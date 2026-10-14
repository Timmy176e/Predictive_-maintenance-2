# Predictive Maintenance Dashboard

A Streamlit app: upload a CSV of current asset readings, get back a
recommended maintenance type, failure risk score, remaining useful life, and
predicted next maintenance due date per asset.

Models are Random Forests trained on a synthetic dataset built to mirror this
plant's real shift maintenance reports - retrain on real historical
maintenance records once they're available, re-upload the new `.joblib`
files to Google Drive, and swap in their file IDs in `app.py` (see below).

## Folder contents

```
app.py              - the Streamlit app (downloads the models from Drive on first run)
requirements.txt    - exact package versions used to train the models
sample_input.csv    - a ready-made file to test the upload with
```

There's no `models/` folder here on purpose - the 4 trained models (~27MB
total) are hosted on Google Drive instead of committed to Git, since GitHub's
web uploader rejects files that large. `app.py` downloads them into a local
`models/` folder automatically the first time it runs, using these file IDs:

| Model | Google Drive file |
|---|---|
| `maintenance_type_classifier.joblib` | `1yF-zgkKA_fX9qPkVeMRuAL6N9EafVlwL` |
| `failure_risk_regressor.joblib` | `1oNpbkDPf8W5LlWumroQlBWmHgiy6K0mo` |
| `rul_regressor.joblib` | `1BjAYSTWLqKSktTWri1mNzxSIF0mMHbcW` |
| `days_until_due_regressor.joblib` | `1wqk3O2bw_cTIZQw9leNhoTQ3NSv4ZxV9` |

**Before deploying, double-check all 4 files' sharing setting is "Anyone with
the link" (Viewer).** If any one of them is still private, the download will
fail (or silently save an HTML "permission denied" page instead of the real
model, which then fails to load) - this is the single most common thing that
goes wrong with this setup, so check it first if the app errors out.

If you ever re-upload a retrained model to Drive, get its new file ID from
its share link (the part between `/d/` and `/view`) and swap it into the
`MODEL_FILES` dictionary near the top of `app.py`.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

First launch will show a "Loading models..." spinner while it pulls the 4
files from Drive (needs internet access); after that they're cached in the
local `models/` folder so it won't re-download on every rerun. It opens at
`http://localhost:8501` - upload `sample_input.csv` to try it immediately, or
any CSV with these columns:

- **Required:** `age_days`, `fault_count_last_30d`, `fault_count_last_90d`,
  `downtime_minutes_last_90d`, `days_since_last_maintenance`,
  `degradation_index`, `comp_type`, `section`, `last_maintenance_type`, `line`
- **Optional:** `component_id`, `area_code`, `component_name`,
  `snapshot_date` (defaults to today if left out)

## Deploy it for free (Streamlit Community Cloud)

1. Create a new GitHub repository and push just these 3 files (`app.py`,
   `requirements.txt`, `sample_input.csv`) - no large files this time, so the
   regular GitHub web uploader or a plain `git push` both work fine:
   ```bash
   git init
   git add app.py requirements.txt sample_input.csv
   git commit -m "Predictive maintenance dashboard"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **share.streamlit.io**, sign in with GitHub, and click **New app**.
3. Pick the repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`, the app
   downloads the models from Drive on first boot, and you get a public
   `https://<something>.streamlit.app` URL to share or open in your pitch.
5. Any time you push a new commit (e.g. a new `MODEL_FILES` id after
   retraining), the app redeploys automatically.

If your company would rather this not be public once real data is involved,
Streamlit Community Cloud apps can be restricted to specific viewers under
the app's sharing settings, or the app can instead be run on an internal
server with the same two commands under "Run it locally" - and the model
files should move off a public Drive link onto something access-controlled
at that point too.
