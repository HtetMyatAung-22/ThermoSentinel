---
title: ThermoSentinel v0.2
emoji: 🛰️
colorFrom: blue
colorTo: orange
sdk: gradio
app_file: app.py
pinned: false
---

# ThermoSentinel v0.2

Conference-ready prototype integrating **live NASA FIRMS NOAA-20/NOAA-21 NRT evidence** with a **simulated predictive ground-sensor layer**.

## New in v0.2

- Live NASA FIRMS Area API query
- Boundary-distance calculation for NRT VIIRS detections
- 10-minute API cache
- Separate LIVE versus SIMULATED evidence labels
- One-click conference scenarios:
  - Routine
  - Developing Hotspot
  - Thermal Escalation
- Redesigned technician-facing main screen
- 6 h / 12 h / 24 h thermal-escalation forecast
- Explainable prototype risk contributions
- Technician feedback logging

## NASA FIRMS key

Request a free MAP_KEY from the official NASA FIRMS Map Key page.

### Google Colab

The included notebook securely asks for the key using `getpass()` and stores it only in the runtime environment:

```python
from getpass import getpass
import os
os.environ["FIRMS_MAP_KEY"] = getpass("NASA FIRMS MAP_KEY: ")
```

### Hugging Face Spaces

Create a secret named exactly:

`FIRMS_MAP_KEY`

Do not hard-code or publicly upload the key.

## Scientific wording

The system is **not a validated fire predictor**.

Use:
- “thermal-escalation forecast”
- “prototype predictive screening”
- “near-real-time VIIRS evidence”
- “No NRT VIIRS detection observed”

Do not use:
- “AI predicts a fire tomorrow”
- “No satellite detection means no fire”
- “validated early-warning system”

The Routine / Watch / Investigate logic remains preliminary until it is prospectively validated with real ground sensors and technician-confirmed events.
