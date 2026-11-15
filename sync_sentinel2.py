"""ThermoSentinel: dry-run-first, scene-based Sentinel-2 indices sync.
Google authentication uses keyless GitHub Actions Workload Identity Federation
and Application Default Credentials (ADC); no JSON private key needed.
"""
import datetime as dt
import json
import math
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

import ee

SCHEMA_VARS = ('NDVI', 'NDWI', 'BSI')
BOUNDARY = 'projects/gee-uav-thermal-imaging/assets/Site_Boundary'
START_DAYS_BACK = int(os.getenv('LOOKBACK_DAYS', '14'))
MIN_COVERAGE = 0.75
DRY_RUN = os.getenv('DRY_RUN', 'true').lower() != 'false'


def initialize_ee():
    project = os.environ['EE_PROJECT_ID']
    # google-github-actions/auth provides GOOGLE_APPLICATION_CREDENTIALS.
    # ADC loads the short-lived, federated identity; no private key is stored.
    import google.auth
    credentials, _ = google.auth.default(scopes=[
        'https://www.googleapis.com/auth/earthengine',
        'https://www.googleapis.com/auth/cloud-platform',
    ])
    ee.Initialize(credentials=credentials, project=project)


def build_collection(start, end):
    site = ee.FeatureCollection(BOUNDARY).geometry()
    area = ee.Image.pixelArea().reduceRegion(
        reducer=ee.Reducer.sum(), geometry=site, scale=20,
        crs='EPSG:32647', maxPixels=100000000).getNumber('area')

    def calculate(im):
        scl = im.select('SCL')
        clear = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
        blue = im.select('B2').multiply(.0001)
        green = im.select('B3').multiply(.0001)
        red = im.select('B4').multiply(.0001)
        nir = im.select('B8').multiply(.0001)
        swir = im.select('B11').multiply(.0001)
        ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
        ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
        bsi = swir.add(red).subtract(nir.add(blue)).divide(
            swir.add(red).add(nir).add(blue)).rename('BSI')
        indices = ee.Image.cat([ndvi, ndwi, bsi]).updateMask(clear).updateMask(im.select('B11').mask())
        valid_area = ee.Image.pixelArea().updateMask(indices.select('NDVI').mask()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=site, scale=20,
            crs='EPSG:32647', maxPixels=100000000).get('area')
        coverage = ee.Number(ee.Algorithms.If(valid_area, valid_area, 0)).divide(area)
        values = indices.reduceRegion(reducer=ee.Reducer.median(), geometry=site,
                                      scale=20, crs='EPSG:32647', maxPixels=100000000)
        return im.set('coverage_fraction', coverage).set('vNDVI', values.get('NDVI')).set(
            'vNDWI', values.get('NDWI')).set('vBSI', values.get('BSI'))

    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(site).filterDate(start, end).map(calculate)
                  .filter(ee.Filter.gte('coverage_fraction', MIN_COVERAGE))
                  .filter(ee.Filter.notNull(['vNDVI', 'vNDWI', 'vBSI'])))
    return collection


def extract_rows(collection):
    count = collection.size().getInfo()
    if count > 100:
        raise RuntimeError(f'Too many scenes for one execution ({count}); shorten lookback')
    if not count:
        return []
    images = collection.toList(count)
    # Retrieve only selected metadata. No imagery is downloaded.
    summaries = ee.List.sequence(0, count-1).map(lambda i: ee.Dictionary({
        'scene_id': ee.Image(images.get(i)).get('system:index'),
        'time': ee.Image(images.get(i)).get('system:time_start'),
        'coverage': ee.Image(images.get(i)).get('coverage_fraction'),
        'NDVI': ee.Image(images.get(i)).get('vNDVI'),
        'NDWI': ee.Image(images.get(i)).get('vNDWI'),
        'BSI': ee.Image(images.get(i)).get('vBSI'),
    })).getInfo()
    rows = []
    for scene in summaries:
        timestamp = dt.datetime.fromtimestamp(scene['time']/1000, dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        cov = float(scene['coverage'])
        if not (math.isfinite(cov) and MIN_COVERAGE <= cov <= 1.00001):
            raise ValueError('Invalid coverage for '+str(scene['scene_id']))
        for variable in SCHEMA_VARS:
            val = float(scene[variable])
            if not math.isfinite(val) or not -1.00001 <= val <= 1.00001:
                raise ValueError(f'Invalid {variable} in scene {scene["scene_id"]}')
            rows.append({
                'record_id': f'PSM_S2_{variable}_{scene["scene_id"]}',
                'observed_at': timestamp,
                'site_id': 'PSM',
                'satellite_source': 'Sentinel-2 SR Harmonized',
                'variable': variable,
                'value': round(val, 8),
                'spatial_support': '22.49 ha PSM site polygon; SCL classes 4/5/6; site median at 20 m analysis scale',
                'valid_coverage_fraction': min(1.0, round(cov, 8)),
                'quality_status': 'review',
                'notes': 'Automated, provisional. NDWI is McFeeters green/NIR; verify against thesis definition. Scene '+str(scene['scene_id']),
                'source_file': 'GEE_AUTOMATED_S2_SR_HARMONIZED',
            })
    return rows


def upload_new_records(rows):
    if not rows:
        print('No qualifying new-window scenes. No write needed.')
        return
    if DRY_RUN:
        print('DRY RUN: no Supabase writes. Candidate observations: '+str(len(rows)))
        print(json.dumps(rows[:3], indent=2))
        return
    url = os.environ['SUPABASE_URL'].rstrip('/')
    if not url.endswith('.supabase.co'):
        raise ValueError('SUPABASE_URL must be the project base URL ending .supabase.co')
    key = os.environ['SUPABASE_SECRET_KEY']
    endpoint = url + '/rest/v1/satellite_observations?on_conflict=record_id'
    # PostgREST ignores existing record IDs, leaving researcher-reviewed records unchanged.
    req = urllib.request.Request(endpoint,
        data=json.dumps(rows, allow_nan=False).encode('utf-8'),
        headers={'apikey': key,
                 'Content-Type': 'application/json',
                 'Prefer': 'resolution=ignore-duplicates,return=minimal'},
        method='POST')
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            print(f'Supabase insert-if-new request accepted: HTTP {resp.status}; submitted {len(rows)} candidate rows')
    except urllib.error.HTTPError as e:
        # Do not print URL, headers or secret; response body can sometimes contain row values.
        print(f'Supabase rejected write: HTTP {e.code}', file=sys.stderr)
        raise


def main():
    now = dt.datetime.now(dt.timezone.utc)
    start = (now-dt.timedelta(days=START_DAYS_BACK)).date().isoformat()
    # EE end date is exclusive; extra future day is harmless and covers today's observations if available.
    end = (now+dt.timedelta(days=1)).date().isoformat()
    print(f'Mode: {"DRY RUN" if DRY_RUN else "PUBLISH"}; UTC window: {start} to {end} (end exclusive)')
    initialize_ee()
    print('Earth Engine service-account authentication succeeded')
    rows = extract_rows(build_collection(start, end))
    print(f'Qualifying scene-index observations: {len(rows)}; scenes: {len(rows)//3}')
    upload_new_records(rows)


if __name__ == '__main__':
    main()
