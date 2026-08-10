"""
Vercel serverless function — German rent prediction.

Loads the scikit-learn pipeline (model/housing_model.pkl) once per warm
container and serves predictions over POST /api/predict.

Request  JSON: {"livingSpace":60,"noRooms":3,"yearConstructed":2015,
                "state":"Berlin","heatingType":"central_heating",
                "balcony":true,"newlyConst":false}
Response JSON: {"prediction":1234.56,"r2":0.71,"currency":"EUR"}
"""

import json
import os
import traceback
from http.server import BaseHTTPRequestHandler

import joblib
import pandas as pd

# ---------------------------------------------------------------- paths
# The function runs from the deployment root, but be defensive about cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model", "housing_model.pkl")
METRICS_PATH = os.path.join(ROOT, "model", "metrics.json")

# ------------------------------------------------------- lazy singletons
_MODEL = None
_METRICS = {"r2_score": 0}
_LOAD_ERROR = None

# Column order must match train_model.py exactly.
FEATURES = [
    "livingSpace",
    "noRooms",
    "heatingType",
    "balcony",
    "newlyConst",
    "yearConstructed",
    "state",
]

STATES = [
    "Nordrhein_Westfalen", "Sachsen", "Bremen", "Bayern", "Berlin",
    "Hessen", "Hamburg", "Baden_Wuerttemberg", "Thueringen",
    "Sachsen_Anhalt", "Other",
]

HEATING = [
    "central_heating", "floor_heating", "district_heating",
    "gas_heating", "oil_heating", "self_contained_central_heating",
]


def get_model():
    """Load the pipeline once and cache it for the container's lifetime."""
    global _MODEL, _METRICS, _LOAD_ERROR
    if _MODEL is not None or _LOAD_ERROR is not None:
        return _MODEL

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "housing_model.pkl not found at %s. Confirm vercel.json "
                "includeFiles covers model/**." % MODEL_PATH
            )
        _MODEL = joblib.load(MODEL_PATH)

        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as fh:
                _METRICS = json.load(fh)
    except Exception as exc:  # noqa: BLE001 - surfaced to the client
        _LOAD_ERROR = "%s: %s" % (type(exc).__name__, exc)
        traceback.print_exc()

    return _MODEL


def clamp(value, low, high, default):
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if num != num:  # NaN
        return default
    return max(low, min(high, num))


def predict(payload):
    """Validate input, run the pipeline, return a JSON-ready dict."""
    model = get_model()
    if model is None:
        return 500, {"error": "Model unavailable", "detail": _LOAD_ERROR}

    state = payload.get("state")
    if state not in STATES:
        state = "Other"

    heating = payload.get("heatingType")
    if heating not in HEATING:
        heating = "central_heating"

    row = {
        "livingSpace": clamp(payload.get("livingSpace"), 10, 500, 60.0),
        "noRooms": clamp(payload.get("noRooms"), 1, 10, 3.0),
        "heatingType": heating,
        "balcony": 1.0 if payload.get("balcony") else 0.0,
        "newlyConst": 1.0 if payload.get("newlyConst") else 0.0,
        "yearConstructed": clamp(payload.get("yearConstructed"), 1900, 2026, 2015.0),
        "state": state,
    }

    frame = pd.DataFrame([row], columns=FEATURES)

    try:
        value = float(model.predict(frame)[0])
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return 500, {"error": "Prediction failed", "detail": str(exc)}

    return 200, {
        "prediction": round(value, 2),
        "currency": "EUR",
        "r2": _METRICS.get("r2_score", 0),
        "echo": row,
    }


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel requires this name
    def _send(self, status, body):
        raw = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        """Health check — confirms the pickle loads without a full predict."""
        model = get_model()
        self._send(200 if model is not None else 500, {
            "status": "ok" if model is not None else "error",
            "modelLoaded": model is not None,
            "detail": _LOAD_ERROR,
            "r2": _METRICS.get("r2_score", 0),
            "states": STATES,
            "heatingTypes": HEATING,
        })

    def do_POST(self):  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return self._send(400, {"error": "Invalid JSON body"})

        status, body = predict(payload)
        self._send(status, body)
