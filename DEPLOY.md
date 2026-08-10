# Deploying the German Rent Predictor to Vercel

Your trained `model/housing_model.pkl` is used as-is. Nothing is retrained,
and the prediction numbers are identical to the Streamlit app.

## What goes where

Copy these into the root of `snehal201/Germany_Hause_Price_Prediction_App`:

```
Germany_Hause_Price_Prediction_App/
├── api/
│   └── predict.py              ← NEW  serverless prediction endpoint
├── model/
│   ├── housing_model.pkl       ← already there, untouched
│   └── metrics.json            ← already there, untouched
├── index.html                  ← NEW  frontend
├── vercel.json                 ← NEW  runtime config
├── .vercelignore               ← NEW  keeps the 61 MB CSV out of the bundle
├── requirements.txt            ← REPLACES the old one (slim, runtime only)
├── requirements-streamlit.txt  ← NEW  your original deps, for local dev
├── app.py                      ← keep, still runs locally
└── train_model.py              ← keep, still runs locally
```

`app.py` and `train_model.py` keep working locally:

```bash
pip install -r requirements-streamlit.txt
streamlit run app.py
```

## Step 1 — pin your scikit-learn version

This is the one thing most likely to break the deploy. The pickle must be
unpickled by the same scikit-learn version that created it.

```bash
python -c "import sklearn, numpy, pandas; print(sklearn.__version__, numpy.__version__, pandas.__version__)"
```

Put those exact versions in `requirements.txt`. If they don't match, the
function usually still loads but may emit `InconsistentVersionWarning` — or
fail outright on a major version gap.

## Step 2 — push to GitHub

```bash
cd Germany_Hause_Price_Prediction_App
git add api/predict.py index.html vercel.json .vercelignore \
        requirements.txt requirements-streamlit.txt DEPLOY.md
git commit -m "Add Vercel serverless deployment"
git push
```

## Step 3 — import into Vercel

1. vercel.com → **Add New → Project**
2. Import `Germany_Hause_Price_Prediction_App`
3. Framework Preset: **Other**. Leave build command and output directory empty.
4. **Deploy**

## Step 4 — verify

Open `https://<your-project>.vercel.app/api/predict` in a browser. A healthy
deployment returns:

```json
{"status":"ok","modelLoaded":true,"r2":0.71, ...}
```

If `modelLoaded` is `false`, the `detail` field says why. Then open the root
URL — the badge in the header shows **Model ready** when the pickle loaded.

## Known limits on the free Hobby plan

| Limit | Value | Impact here |
|---|---|---|
| Function duration | 10 s | Cold start ≈ 3–6 s. Fits, but not comfortably. |
| Python bundle | 500 MB uncompressed | sklearn + pandas + numpy ≈ 420 MB. Tight. |
| Invocations | 1 M / month | Not a concern. |
| Commercial use | Not allowed | Portfolio use is fine. |

The page pings `/api/predict` on load to warm the container, so the first
real click is usually fast.

### If the bundle exceeds 500 MB

Drop pandas and feed the pipeline a numpy array instead — but the fitted
`ColumnTransformer` refers to columns by name, so this needs the transformer
rebuilt with positional indices in `train_model.py`.

### If cold starts annoy you

Convert the 50-tree forest to JSON and run inference in the browser. Zero
cold start, zero size limit, identical output. Ask and I'll build it — I just
need `housing_model.pkl` uploaded.
