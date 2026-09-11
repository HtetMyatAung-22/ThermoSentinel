# ThermoSentinel v0.2 — Render Deployment Guide

This package is prepared for deployment as a Python web service on Render.

## 1. Create a GitHub repository

Create a new public or private repository, for example:

`ThermoSentinel`

Upload the **contents of this folder**, not the ZIP itself.

The repository root should contain:

- `app.py`
- `requirements.txt`
- `README.md`
- `render.yaml`
- `Procfile`
- `Conference_Demo_Script.md`
- `data/`

`app.py` must be at the repository root.

## 2. Create a Render Web Service

In Render:

1. Choose **New → Web Service**
2. Connect your GitHub account
3. Select the ThermoSentinel repository
4. Render should detect the Python app automatically

If you configure it manually:

- **Runtime:** Python
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `python app.py`

The app reads Render's `PORT` environment variable automatically.

## 3. Add the NASA FIRMS key

In the Render service:

**Environment → Add Environment Variable**

Use:

- Key: `FIRMS_MAP_KEY`
- Value: your private NASA FIRMS MAP_KEY

Do not put the key in `app.py`, GitHub, README, PowerPoint, or screenshots.

## 4. Deploy

Start the deployment and wait until the service becomes live.

Open the Render URL and test:

1. Conference Demo
2. Routine
3. Developing Hotspot
4. Thermal Escalation
5. Live NASA FIRMS
6. Refresh NASA FIRMS now
7. Technician Feedback

## 5. Conference use

Open the app several minutes before presenting so the service is awake and ready.

Recommended demonstration sequence:

**Routine → Developing Hotspot → Thermal Escalation → Live NASA FIRMS**

Use this wording:

> “The live satellite layer is real NASA FIRMS near-real-time evidence. The ground-sensor and 6/12/24-hour predictive layers are currently simulated prototype outputs pending field validation.”

## Important scientific wording

Correct:
- prototype thermal-escalation forecast
- simulated ground-sensor prediction
- live NASA FIRMS evidence
- boundary-aware monitoring
- prospective field validation required

Avoid:
- AI predicts a fire tomorrow
- validated early-warning system
- no FIRMS detection means no fire
- 80% chance of fire

## Persistent storage note

The current technician feedback log is written to a local CSV. On an ephemeral/free cloud instance, this is not a permanent database. For the conference prototype that is acceptable. For field deployment, use a persistent database.
