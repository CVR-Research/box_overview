# box_overview
* Code is unmaintained / work in progress

## Live box-spread stack

Directory `live_box_spreads/` contains a **low-latency Alpaca market data stack** that builds box-spread books in real time and renders a visual CV-style dashboard:

```
live_box_spreads/
├── boxspread.py          # immutable spread math model
├── alpaca_client.py      # REST API wrapper (quotes + chains)
├── alpaca_stream.py      # websocket client (msgpack)
├── ingest.py             # snapshot collector + CLI
├── config.yaml           # universe + thresholds
├── data/live_snapshots   # parquet/csv snapshots
├── cv.yaml               # editable visual CV content
└── dashboard/app.py      # Plotly Dash analytics UI
```

### Prereqs
- Python 3.10+
- `pip install -r requirements.txt` (Dash/Plotly plus pandas, **Polars** for fast data prep, PyArrow for parquet IO)
- Environment variables `ALPACA_API_KEY` and `ALPACA_API_SECRET`.
- Optional: `ALPACA_DATA_BASE_URL` / `ALPACA_DATA_FEED` for REST, `ALPACA_STREAM_URL` for websocket.

### Quick start helper (Conda + launcher)
```
cd live_box_spreads
./manage.sh
```
The script will:
1. Ensure `conda` exists, create/update the `box_live` env (Python 3.11), and install `requirements.txt`.
2. Load `live_box_spreads/.env` if present and verify `ALPACA_API_KEY`/`ALPACA_API_SECRET`.
3. Prompt you to (a) take a single snapshot, (b) run the continuous collector, (c) launch the Dash dashboard, or (d) run both collector and dashboard together (collector in the background until you exit).

### Configure credentials
Create or edit `live_box_spreads/.env`:
```
ALPACA_API_KEY=''
ALPACA_API_SECRET=''
```

### Manual data collection (if you prefer)
```
cd live_box_spreads
python ingest.py --loop            # continuous polling per config interval
# or run once for a single snapshot
python ingest.py
```
The collector uses Polars for the heavy data wrangling and writes timestamped parquet files under `live_box_spreads/data/live_snapshots/`, pruning the folder to the configured retention depth.

### Dashboard (snapshot mode)
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

### Dashboard (stream mode)
Use Alpaca websocket msgpack stream to refresh the dashboard in near real time:
```
cd live_box_spreads/dashboard
DASH_USE_STREAM=1 DASH_UPDATE_MS=1000 python app.py
```
Stream mode pulls live option quotes via websocket, builds spreads in memory, and updates the dashboard every refresh tick. Adjust `DASH_UPDATE_MS` for tighter or looser update cadence.

### Visual CV content
Edit `live_box_spreads/cv.yaml` to update the header, metrics, skills, and projects displayed in the dashboard.

### Legacy snapshot scripts
The previous CSV snapshot utilities now live under `archive_snapshot_pipeline/` unchanged for reference.
