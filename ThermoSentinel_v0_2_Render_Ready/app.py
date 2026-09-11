
from pathlib import Path
import os, json, io, math, time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# ------------------------ Load project data ------------------------
with open(DATA / "site_config.json", encoding="utf-8") as f:
    CONFIG = json.load(f)

SITE_NAME = CONFIG.get("site_name", "Open dumpsite")
SITE_LAT = float(CONFIG["site_lat"])
SITE_LON = float(CONFIG["site_lon"])

with open(DATA / "site_boundary.geojson", encoding="utf-8") as f:
    BOUNDARY = json.load(f)

zones = pd.read_csv(DATA / "zone_config.csv")
sensors = pd.read_csv(DATA / "simulated_sensor_stream.csv", parse_dates=["timestamp"])
thermal = pd.read_csv(DATA / "landsat_thermal.csv", parse_dates=["date"])

for c in ["delta_P95_C","robust_Z","pixel_coverage_fraction"]:
    if c in thermal.columns:
        thermal[c] = pd.to_numeric(thermal[c], errors="coerce")
thermal["good_pixel_coverage"] = thermal["good_pixel_coverage"].astype(str).str.lower().eq("true")

# ------------------------ General geometry helpers ------------------------
def boundary_coords():
    geom = BOUNDARY["features"][0]["geometry"]
    coords = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
    return [(float(p[0]), float(p[1])) for p in coords]

BOUNDARY_RING = boundary_coords()

def circle_coords(lat, lon, radius_m, n=120):
    ang = np.linspace(0, 2*np.pi, n)
    dlat = (radius_m / 111320.0) * np.sin(ang)
    dlon = (radius_m / (111320.0 * np.cos(np.radians(lat)))) * np.cos(ang)
    return lat + dlat, lon + dlon

def point_in_polygon(lon, lat, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        hit = ((yi > lat) != (yj > lat)) and (
            lon < (xj-xi)*(lat-yi)/(yj-yi+1e-15) + xi
        )
        if hit:
            inside = not inside
        j = i
    return inside

def local_xy(lon, lat, lon0=SITE_LON, lat0=SITE_LAT):
    x = (lon-lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (lat-lat0) * 110540.0
    return x, y

def point_segment_distance(px, py, ax, ay, bx, by):
    vx, vy = bx-ax, by-ay
    wx, wy = px-ax, py-ay
    c1 = vx*wx + vy*wy
    if c1 <= 0:
        return math.hypot(px-ax, py-ay)
    c2 = vx*vx + vy*vy
    if c2 <= c1:
        return math.hypot(px-bx, py-by)
    t = c1/c2
    qx, qy = ax+t*vx, ay+t*vy
    return math.hypot(px-qx, py-qy)

def distance_to_boundary_m(lon, lat):
    if point_in_polygon(lon, lat, BOUNDARY_RING):
        return 0.0
    px, py = local_xy(lon, lat)
    pts = [local_xy(x,y) for x,y in BOUNDARY_RING]
    d = float("inf")
    for i in range(len(pts)-1):
        d = min(d, point_segment_distance(px, py, *pts[i], *pts[i+1]))
    return d

# ------------------------ Simulation / forecast layer ------------------------
def current_window(sim_index=None):
    if sim_index is None:
        sim_index = 0
    unique_times = sensors["timestamp"].drop_duplicates().sort_values().reset_index(drop=True)
    idx = max(0, min(len(unique_times)-1, len(unique_times)-1-int(sim_index)))
    current_ts = unique_times.iloc[idx]
    return current_ts, sensors[sensors["timestamp"] <= current_ts].copy()

def slope_per_hour(df, col, hours=3):
    d = df.sort_values("timestamp")
    end = d["timestamp"].max()
    w = d[d["timestamp"] >= end-pd.Timedelta(hours=hours)]
    if len(w) < 3:
        return 0.0
    x = (w["timestamp"]-w["timestamp"].min()).dt.total_seconds().to_numpy()/3600
    y = pd.to_numeric(w[col], errors="coerce").to_numpy()
    mask = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[mask], y[mask], 1)[0]) if mask.sum() >= 3 else 0.0

def logistic(x):
    return 1/(1+np.exp(-x))

def zone_features(history, zone):
    z = history[history["zone_id"] == zone].sort_values("timestamp")
    latest = z.iloc[-1]
    t_slope = slope_per_hour(z, "temp_C", 3)
    co_slope = slope_per_hour(z, "CO_ppm", 3)
    ratio = latest["CH4_pct"]/max(latest["CO2_pct"], 0.01)
    old = z.iloc[max(0, len(z)-19)]
    ratio_old = old["CH4_pct"]/max(old["CO2_pct"], 0.01)
    rain24 = z[z["timestamp"] >= z["timestamp"].max()-pd.Timedelta(hours=24)]["rain_mm"].sum()
    return {
        "temp":float(latest["temp_C"]), "co":float(latest["CO_ppm"]),
        "ch4":float(latest["CH4_pct"]), "co2":float(latest["CO2_pct"]),
        "o2":float(latest["O2_pct"]), "air":float(latest["air_temp_C"]),
        "rh":float(latest["RH_pct"]), "thermal_index":float(latest["thermal_index"]),
        "t_slope":t_slope, "co_slope":co_slope,
        "ratio_change":float(ratio-ratio_old), "rain24":float(rain24)
    }

def forecast_zone(history, zone):
    f = zone_features(history, zone)
    c_temp = np.clip((f["temp"]-42)/12, 0, 1.5)
    c_trend = np.clip(f["t_slope"]/1.5, 0, 1.5)
    c_co = np.clip((f["co"]-12)/35, 0, 1.5)
    c_cotrend = np.clip(f["co_slope"]/6, 0, 1.5)
    c_ratio = np.clip((-f["ratio_change"])/0.08, 0, 1.5)
    c_thermal = np.clip((f["thermal_index"]-0.28)/0.5, 0, 1.5)
    c_dry = 1.0 if f["rain24"] < 0.5 else 0.0

    score = (1.25*c_temp + 1.45*c_trend + 1.05*c_co + 0.75*c_cotrend
             + 0.70*c_ratio + 1.10*c_thermal + 0.20*c_dry - 2.6)
    damp = {6:0.75, 12:0.55, 24:0.30}
    fc, risk = {}, {}
    for h in [6,12,24]:
        pred = f["temp"] + f["t_slope"]*h*damp[h]
        fc[h] = float(np.clip(pred, f["temp"]-3, f["temp"]+18))
        bonus = {6:-0.15,12:0.10,24:0.25}[h]
        risk[h] = float(np.clip(logistic(score+bonus),0.01,0.99))
    contrib = {
        "Temperature trend":1.45*c_trend,
        "Temperature level":1.25*c_temp,
        "Thermal-camera index":1.10*c_thermal,
        "CO level":1.05*c_co,
        "CO trend":0.75*c_cotrend,
        "CH4/CO2 trend":0.70*c_ratio,
        "Dry weather":0.20*c_dry,
    }
    return f, fc, risk, contrib

def status_from_risk(r):
    if r >= 0.70: return "INVESTIGATE", "#d95f02"
    if r >= 0.40: return "WATCH", "#c28a00"
    return "ROUTINE", "#1b8f63"

# ------------------------ Live NASA FIRMS layer ------------------------
FIRMS_CACHE = {"time":0, "data":None, "message":"Not queried"}
CACHE_SECONDS = 600

def query_firms(force=False):
    key = os.getenv("FIRMS_MAP_KEY", "").strip()
    if not key:
        return pd.DataFrame(), "FIRMS_MAP_KEY is not configured. Live satellite panel is in OFFLINE mode."

    now = time.time()
    if (not force and FIRMS_CACHE["data"] is not None
        and now-FIRMS_CACHE["time"] < CACHE_SECONDS):
        return FIRMS_CACHE["data"].copy(), FIRMS_CACHE["message"]

    # about +/- 6 km around the site
    lat_pad = 0.055
    lon_pad = 0.055 / max(math.cos(math.radians(SITE_LAT)), 0.2)
    bbox = f"{SITE_LON-lon_pad:.5f},{SITE_LAT-lat_pad:.5f},{SITE_LON+lon_pad:.5f},{SITE_LAT+lat_pad:.5f}"

    sources = ["VIIRS_NOAA20_NRT","VIIRS_NOAA21_NRT"]
    frames, errors = [], []
    for src in sources:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{src}/{bbox}/1"
        try:
            r = requests.get(url, timeout=25)
            r.raise_for_status()
            txt = r.text.strip()
            if not txt or "latitude" not in txt.splitlines()[0].lower():
                errors.append(f"{src}: unexpected API response")
                continue
            d = pd.read_csv(io.StringIO(txt))
            if len(d):
                d["source_product"] = src
                frames.append(d)
        except Exception as e:
            errors.append(f"{src}: {type(e).__name__}")

    if frames:
        d = pd.concat(frames, ignore_index=True)
        d["latitude"] = pd.to_numeric(d["latitude"], errors="coerce")
        d["longitude"] = pd.to_numeric(d["longitude"], errors="coerce")
        d["frp"] = pd.to_numeric(d.get("frp"), errors="coerce")
        d = d.dropna(subset=["latitude","longitude"]).copy()
        d["boundary_distance_m"] = [
            distance_to_boundary_m(lon, lat) for lon,lat in zip(d["longitude"],d["latitude"])
        ]
        if "acq_date" in d and "acq_time" in d:
            hhmm = d["acq_time"].astype(str).str.replace(r"\.0$","",regex=True).str.zfill(4)
            dt = pd.to_datetime(
                d["acq_date"].astype(str)+" "+hhmm.str[:2]+":"+hhmm.str[2:4],
                errors="coerce", utc=True
            )
            d["acq_datetime_utc"] = dt
            now_utc = pd.Timestamp.now(tz="UTC")
            d["age_hours"] = (now_utc-dt).dt.total_seconds()/3600
        d = d.sort_values("boundary_distance_m")
    else:
        d = pd.DataFrame()

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"NASA FIRMS queried {stamp}. NOAA-20 + NOAA-21, most recent 1-day feed."
    if errors:
        msg += " " + "; ".join(errors)
    FIRMS_CACHE.update(time=now, data=d.copy(), message=msg)
    return d, msg

def live_firms_summary(force=False):
    d, msg = query_firms(force=force)
    if d.empty:
        headline = "No NRT VIIRS detection observed in the query area."
        rows = pd.DataFrame(columns=["satellite","acq_date","acq_time","frp","boundary_distance_m","confidence","daynight"])
        counts = {"500 m":0,"1 km":0,"2 km":0,"5 km":0}
        nearest = "—"
    else:
        counts = {
            "500 m":int((d["boundary_distance_m"]<=500).sum()),
            "1 km":int((d["boundary_distance_m"]<=1000).sum()),
            "2 km":int((d["boundary_distance_m"]<=2000).sum()),
            "5 km":int((d["boundary_distance_m"]<=5000).sum()),
        }
        near = d.iloc[0]
        nearest = f"{near['boundary_distance_m']:.0f} m"
        headline = f"{len(d)} NRT VIIRS detection(s) in the ~6 km query area."
        cols = [c for c in ["satellite","acq_date","acq_time","frp","boundary_distance_m","confidence","daynight","source_product"] if c in d.columns]
        rows = d[cols].head(20).copy()
        if "boundary_distance_m" in rows:
            rows["boundary_distance_m"] = rows["boundary_distance_m"].round(0)

    html = f"""
    <div class="satbox">
      <h3 style="margin-top:0;">🛰 Live NASA FIRMS status</h3>
      <div style="font-size:18px;font-weight:700;">{headline}</div>
      <div style="margin-top:8px;">
        <b>≤500 m:</b> {counts['500 m']} &nbsp; | &nbsp;
        <b>≤1 km:</b> {counts['1 km']} &nbsp; | &nbsp;
        <b>≤2 km:</b> {counts['2 km']} &nbsp; | &nbsp;
        <b>≤5 km:</b> {counts['5 km']} &nbsp; | &nbsp;
        <b>Nearest boundary distance:</b> {nearest}
      </div>
      <div style="margin-top:8px;font-size:13px;color:#5d6670;">{msg}</div>
      <div style="margin-top:8px;font-size:13px;"><b>Interpretation:</b> No NRT VIIRS detection observed does not mean no fire.</div>
    </div>
    """
    return html, rows, d

# ------------------------ Visuals ------------------------
def zone_map(history, live=None):
    fig = go.Figure()
    blon=[p[0] for p in BOUNDARY_RING]; blat=[p[1] for p in BOUNDARY_RING]
    fig.add_trace(go.Scattermap(
        lat=blat, lon=blon, mode="lines", fill="toself",
        line=dict(width=2), fillcolor="rgba(255,122,0,0.10)", name="Site boundary"
    ))
    for radius,label in [(500,"500 m"),(1000,"1 km"),(2000,"2 km")]:
        latc, lonc = circle_coords(SITE_LAT,SITE_LON,radius)
        fig.add_trace(go.Scattermap(lat=latc,lon=lonc,mode="lines",name=label,line=dict(width=1)))
    cols, hovers = [], []
    for _, row in zones.iterrows():
        _,_,risk,_ = forecast_zone(history,row.zone_id)
        s,c = status_from_risk(risk[12])
        cols.append(c)
        hovers.append(f"Zone {row.zone_id}<br>{s}<br>12 h risk {risk[12]*100:.0f}%")
    fig.add_trace(go.Scattermap(
        lat=zones["latitude"],lon=zones["longitude"],mode="markers+text",
        text=zones["zone_id"],textposition="top center",marker=dict(size=22,color=cols),
        customdata=np.array(hovers)[:,None],
        hovertemplate="%{customdata[0]}<extra></extra>",name="Monitoring zones"
    ))
    if live is not None and len(live):
        fig.add_trace(go.Scattermap(
            lat=live["latitude"],lon=live["longitude"],mode="markers",
            marker=dict(size=13),name="Live VIIRS NRT",
            customdata=np.column_stack([
                live["boundary_distance_m"].round(0).astype(str),
                live.get("frp",pd.Series([np.nan]*len(live))).round(2).astype(str)
            ]),
            hovertemplate="VIIRS NRT<br>Boundary distance: %{customdata[0]} m<br>FRP: %{customdata[1]} MW<extra></extra>"
        ))
    fig.update_layout(
        map=dict(style="carto-positron",center=dict(lat=SITE_LAT,lon=SITE_LON),zoom=13.8),
        margin=dict(l=0,r=0,t=45,b=0),height=510,title="Boundary-aware site monitor",
        legend=dict(orientation="h",y=0.01,x=0.01,bgcolor="rgba(255,255,255,.85)")
    )
    return fig

def sensor_plot(history, zone):
    z=history[history["zone_id"]==zone].sort_values("timestamp")
    end=z["timestamp"].max(); z=z[z["timestamp"]>=end-pd.Timedelta(hours=24)]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=z["timestamp"],y=z["temp_C"],name="Waste temperature",mode="lines"))
    fig.add_trace(go.Scatter(x=z["timestamp"],y=z["air_temp_C"],name="Air temperature",mode="lines"))
    fig.update_layout(title=f"Zone {zone}: 24-hour thermal development",yaxis_title="°C",height=340,
                      margin=dict(l=50,r=20,t=50,b=40))
    return fig

def gas_plot(history, zone):
    z=history[history["zone_id"]==zone].sort_values("timestamp")
    end=z["timestamp"].max(); z=z[z["timestamp"]>=end-pd.Timedelta(hours=24)]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=z["timestamp"],y=z["CO_ppm"],name="CO",mode="lines"))
    fig.add_trace(go.Scatter(x=z["timestamp"],y=z["CH4_pct"],name="CH₄",mode="lines",yaxis="y2"))
    fig.update_layout(title=f"Zone {zone}: gas indicators",height=340,
        yaxis=dict(title="CO (ppm)"),yaxis2=dict(title="CH₄ (%)",overlaying="y",side="right"),
        margin=dict(l=50,r=55,t=50,b=40))
    return fig

def forecast_plot(history, zone):
    f,fc,risk,_=forecast_zone(history,zone)
    hrs=[0,6,12,24]; vals=[f["temp"],fc[6],fc[12],fc[24]]
    low=[vals[0],vals[1]-1.5,vals[2]-2.5,vals[3]-4]
    high=[vals[0],vals[1]+1.5,vals[2]+2.5,vals[3]+4]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=hrs,y=high,mode="lines",line=dict(width=0),showlegend=False))
    fig.add_trace(go.Scatter(x=hrs,y=low,mode="lines",fill="tonexty",line=dict(width=0),name="Illustrative uncertainty"))
    fig.add_trace(go.Scatter(x=hrs,y=vals,mode="lines+markers",name="Prototype forecast"))
    fig.update_layout(title=f"Zone {zone}: 6 / 12 / 24 h thermal forecast",
        xaxis_title="Forecast horizon (h)",yaxis_title="Temperature (°C)",height=340,
        margin=dict(l=50,r=20,t=50,b=40))
    return fig

def explanation_plot(contrib):
    items=sorted(contrib.items(),key=lambda x:x[1],reverse=True)
    fig=go.Figure(go.Bar(x=[v for _,v in items],y=[k for k,_ in items],orientation="h"))
    fig.update_layout(title="Explainable screening: relative evidence contribution",
        xaxis_title="Relative contribution",height=340,margin=dict(l=150,r=20,t=50,b=40))
    fig.update_yaxes(autorange="reversed")
    return fig

def hero_html(history, zone, scenario_name):
    f,fc,risk,contrib=forecast_zone(history,zone)
    status,color=status_from_risk(risk[12])
    ts=history["timestamp"].max()
    top=sorted(contrib.items(),key=lambda x:x[1],reverse=True)[:3]
    reasons=" • ".join(k for k,_ in top)
    action = (
        f"Inspect Zone {zone} and verify heat, smoke and gas conditions."
        if status=="INVESTIGATE" else
        f"Repeat measurements and review Zone {zone} closely."
        if status=="WATCH" else
        f"Continue routine monitoring of Zone {zone}."
    )
    bars = "".join([
        f"""<div class="riskrow"><b>{h} h</b><div class="riskbar"><div style="width:{risk[h]*100:.0f}%"></div></div><b>{risk[h]*100:.0f}%</b></div>"""
        for h in [6,12,24]
    ])
    return f"""
    <div class="hero">
      <div class="modeflag">CONFERENCE DEMO — SIMULATED GROUND SENSORS</div>
      <div class="hero-grid">
        <div>
          <div class="smallcap">SCENARIO</div><div class="bigtext">{scenario_name}</div>
          <div class="smallcap">HIGHEST FOCUS</div><div class="bigtext">ZONE {zone}</div>
        </div>
        <div>
          <div class="smallcap">CURRENT TEMPERATURE</div><div class="temp">{f['temp']:.1f}°C</div>
          <div>{f['t_slope']:+.2f}°C/h over recent 3 h</div>
        </div>
        <div style="border-left:7px solid {color};padding-left:18px;">
          <div class="smallcap">12-HOUR STATUS</div><div class="status">{status}</div>
          <div class="smallcap">THERMAL-ESCALATION RISK</div>
          {bars}
        </div>
      </div>
      <div class="why"><b>WHY?</b> {reasons}</div>
      <div class="action"><b>TECHNICIAN ACTION:</b> {action}</div>
      <div class="disclaimer">Simulation timestamp: {ts:%Y-%m-%d %H:%M}. Risk values are prototype outputs, not prospectively validated fire probabilities.</div>
    </div>
    """

def landsat_fig():
    d=thermal[thermal["good_pixel_coverage"]].copy()
    fig=go.Figure()
    bg=d[d["thermal_class"].eq("Background range")]
    elev=d[~d["thermal_class"].eq("Background range")]
    fig.add_trace(go.Scatter(x=bg["date"],y=bg["delta_P95_C"],mode="lines+markers",name="Background"))
    if len(elev):
        fig.add_trace(go.Scatter(x=elev["date"],y=elev["delta_P95_C"],mode="markers",
                                 marker=dict(size=10,symbol="diamond"),name="Elevated"))
    fig.update_layout(title="Historical Landsat thermal context",yaxis_title="ΔLST (°C)",
                      height=420,margin=dict(l=50,r=20,t=55,b=40))
    return fig

# ------------------------ Scenario logic ------------------------
SCENARIOS = {
    "Routine": 100,                 # ~16 h before simulated escalation
    "Developing hotspot": 36,       # moderate / WATCH-like
    "Thermal escalation": 0         # INVESTIGATE
}

def scenario_values(name):
    return SCENARIOS[name], "C", name

def update_dashboard(zone, step, scenario_name):
    ts,hist=current_window(step)
    f,fc,risk,contrib=forecast_zone(hist,zone)
    live, _ = query_firms(force=False)
    return (
        hero_html(hist,zone,scenario_name),
        zone_map(hist,live),
        sensor_plot(hist,zone),
        gas_plot(hist,zone),
        forecast_plot(hist,zone),
        explanation_plot(contrib)
    )

def apply_scenario(name):
    step, zone, scenario = scenario_values(name)
    outputs = update_dashboard(zone,step,scenario)
    return step,zone,scenario,*outputs

def refresh_live():
    html, table, live = live_firms_summary(force=True)
    _,hist=current_window(0)
    return html, table, zone_map(hist,live)

# ------------------------ Technician feedback ------------------------
FEEDBACK=ROOT/"technician_feedback.csv"
def save_feedback(zone,condition,temp,notes):
    row=pd.DataFrame([{
        "timestamp":datetime.now().isoformat(timespec="seconds"),
        "zone_id":zone,"condition":condition,"measured_temp_C":temp,"notes":notes
    }])
    if FEEDBACK.exists():
        out=pd.concat([pd.read_csv(FEEDBACK),row],ignore_index=True)
    else:
        out=row
    out.to_csv(FEEDBACK,index=False)
    return f"Saved feedback for Zone {zone}. Total records: {len(out)}",str(FEEDBACK)

# ------------------------ UI ------------------------
CSS = """
.gradio-container {max-width:1500px !important;}
.hero {background:#f8fafc;border:1px solid #dce4eb;border-radius:14px;padding:18px;box-shadow:0 2px 8px rgba(0,0,0,.05);}
.modeflag {display:inline-block;background:#162c3e;color:white;padding:5px 10px;border-radius:999px;font-weight:700;font-size:12px;margin-bottom:14px;}
.hero-grid {display:grid;grid-template-columns:1fr 1fr 1.35fr;gap:22px;align-items:start;}
.smallcap {font-size:12px;font-weight:800;letter-spacing:.06em;color:#62707d;margin-top:4px;}
.bigtext {font-size:24px;font-weight:800;color:#073b5c;margin-bottom:12px;}
.temp {font-size:38px;font-weight:800;color:#073b5c;}
.status {font-size:32px;font-weight:900;color:#073b5c;margin:2px 0 10px;}
.riskrow {display:grid;grid-template-columns:38px 1fr 48px;gap:8px;align-items:center;margin:5px 0;}
.riskbar {height:11px;background:#dfe5ea;border-radius:10px;overflow:hidden;}
.riskbar div {height:100%;background:#0c6f88;}
.why,.action {margin-top:12px;padding:10px 12px;background:white;border-radius:8px;}
.disclaimer {font-size:12px;color:#66717c;margin-top:10px;}
.satbox {background:#f8fafc;border:1px solid #dce4eb;border-radius:12px;padding:16px;}
"""

with gr.Blocks(title="ThermoSentinel v0.2",css=CSS) as demo:
    gr.Markdown("""
# 🛰️ ThermoSentinel v0.2
### Boundary-aware predictive thermal-escalation monitoring

**Two evidence modes are deliberately separated:**  
**LIVE:** NASA FIRMS NOAA-20/NOAA-21 near-real-time satellite observations when `FIRMS_MAP_KEY` is configured.  
**SIMULATED:** 10-minute ground-sensor stream and 6/12/24-hour prototype thermal-escalation forecasts.

The system does **not** claim validated fire prediction.
""")

    with gr.Tab("🎤 Conference Demo"):
        gr.Markdown("### One-click scenarios")
        with gr.Row():
            routine=gr.Button("1 · Routine",variant="secondary")
            developing=gr.Button("2 · Developing Hotspot",variant="secondary")
            escalation=gr.Button("3 · Thermal Escalation",variant="primary")
        with gr.Row():
            scenario=gr.Textbox(value="Thermal escalation",label="Scenario",interactive=False)
            zone=gr.Dropdown(zones["zone_id"].tolist(),value="C",label="Zone")
            step=gr.Slider(0,144,value=0,step=1,label="Simulation rewind (10-minute steps)")
        hero=gr.HTML()
        mapplot=gr.Plot()
        with gr.Row():
            tplot=gr.Plot(); gplot=gr.Plot()
        with gr.Row():
            fplot=gr.Plot(); eplot=gr.Plot()
        refresh=gr.Button("Recalculate")
        main_outputs=[hero,mapplot,tplot,gplot,fplot,eplot]
        refresh.click(update_dashboard,[zone,step,scenario],main_outputs)
        zone.change(update_dashboard,[zone,step,scenario],main_outputs)
        step.change(update_dashboard,[zone,step,scenario],main_outputs)
        demo.load(update_dashboard,[zone,step,scenario],main_outputs)

        routine.click(lambda: apply_scenario("Routine"),None,[step,zone,scenario,*main_outputs])
        developing.click(lambda: apply_scenario("Developing hotspot"),None,[step,zone,scenario,*main_outputs])
        escalation.click(lambda: apply_scenario("Thermal escalation"),None,[step,zone,scenario,*main_outputs])

    with gr.Tab("🛰 Live NASA FIRMS"):
        gr.Markdown("""
### Near-real-time satellite evidence
The API is queried for the latest one-day NOAA-20 and NOAA-21 VIIRS products around the site.
Distances are calculated to the mapped **site boundary**, not only to a reference point.
""")
        live_html=gr.HTML()
        live_table=gr.Dataframe(interactive=False,label="Most recent NRT detections")
        live_map=gr.Plot()
        live_refresh=gr.Button("Refresh NASA FIRMS now",variant="primary")
        live_refresh.click(refresh_live,None,[live_html,live_table,live_map])
        # initial cached query on tab/app load
        def initial_live():
            html,table,live=live_firms_summary(force=False)
            _,hist=current_window(0)
            return html,table,zone_map(hist,live)
        demo.load(initial_live,None,[live_html,live_table,live_map])

    with gr.Tab("🌡 Historical Landsat"):
        gr.Markdown("""
Historical Landsat provides the long-term thermal baseline and seasonal context.
It is **not** presented as a 10-minute real-time feed.
""")
        gr.Plot(value=landsat_fig())

    with gr.Tab("🧑‍🔧 Technician Feedback"):
        gr.Markdown("### Human-in-the-loop confirmation")
        with gr.Row():
            fb_zone=gr.Dropdown(zones["zone_id"].tolist(),value="C",label="Zone")
            condition=gr.Dropdown(
                ["Normal","Elevated heat","Strong odor","Visible smoke","Smoldering",
                 "Open flame","Fire confirmed","Suppression performed"],
                value="Normal",label="Observed condition")
        temp=gr.Number(label="Measured temperature (°C)",value=40)
        notes=gr.Textbox(label="Notes",lines=4)
        save=gr.Button("Save field confirmation")
        msg=gr.Textbox(label="Status")
        outfile=gr.File(label="Feedback log")
        save.click(save_feedback,[fb_zone,condition,temp,notes],[msg,outfile])

    with gr.Tab("🧪 Scientific Status"):
        gr.Markdown("""
### What is real now?
- Historical Landsat thermal database
- Historical VIIRS/FIRMS database
- Mapped site boundary
- Independently verified fire-event records
- Live NASA FIRMS feed when the API key is configured

### What is simulated?
- Zone-level ground sensors
- 6 h / 12 h / 24 h temperature forecasts
- Thermal-escalation probabilities
- Explainability contributions

### What comes next?
Real temperature/weather sensing → gas sensing → thermal camera → prospective model training → technician validation.

**Do not call Routine / Watch / Investigate validated fire-risk classes.**
""")
        gr.Dataframe(value=pd.read_csv(DATA/"sensor_schema.csv"),interactive=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860"))
    )
