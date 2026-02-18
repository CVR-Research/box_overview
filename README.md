# box_overview
-
* Note this is a work in progess

## Live box-spread stack

Directory `live_box_spreads/` now contains everything needed to stream live Tastytrade data, build box-spread books, and visualise lending/borrowing surfaces:

```
live_box_spreads/
├── boxspread.py          # immutable spread math model
├── tastytrade_client.py  # REST API wrapper (login + quotes)
├── ingest.py             # snapshot collector + CLI
├── config.yaml           # universe + thresholds
├── data/live_snapshots   # parquet/csv snapshots
└── dashboard/app.py      # Plotly Dash analytics UI
```

### Prereqs
- Python 3.10+
- `pip install -r requirements.txt` (Dash/Plotly plus pandas, **Polars** for fast data prep, PyArrow for parquet IO)
- Environment variables `TT_USERNAME` and `TT_PASSWORD` with your Tastytrade credentials.

### Quick start helper (Conda + launcher)
```
cd live_box_spreads
./manage.sh
```
The script will:
1. Ensure `conda` exists, create/update the `box_live` env (Python 3.11), and install `requirements.txt`.
2. Verify `TT_USERNAME`/`TT_PASSWORD` are exported in your shell (set them beforehand).
3. Prompt you to (a) take a single snapshot, (b) run the continuous collector, (c) launch the Dash dashboard, or (d) run both collector and dashboard together (collector in the background until you exit).

### Manual data collection (if you prefer)
```
cd live_box_spreads
python ingest.py --loop            # continuous polling per config interval
# or run once for a single snapshot
python ingest.py
```
The collector uses Polars for the heavy data wrangling and writes timestamped parquet files under `live_box_spreads/data/live_snapshots/`, pruning the folder to the configured retention depth.

### Dashboard
```
cd live_box_spreads/dashboard
python app.py
```
The Dash UI auto-refreshes on the same cadence as the collector and exposes:
- Mid/bid implied-rate 3D surfaces (strike × time × rate)
- Volume surface for liquidity pockets
- Term-structure panel with configurable moneyness band
- Skew chart (moneyness vs implied rate) per expiry
- Top lend/borrow opportunities table (shows bid/ask spreads and leg volume)

### Legacy snapshot scripts
The previous CSV snapshot utilities now live under `archive_snapshot_pipeline/` unchanged for reference.
