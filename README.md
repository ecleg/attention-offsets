## Ad-Funded AI Impact Analyzer

With AI being integrated into the curricula of college courses and industry, students must cast aside any reservations about its environmental impact to stay competitive. A platform that leverages the attention of students, as opposed to the dwindling savings of college students, would be able to use advertisements to pay off the damages of prompts. We need to start a discussion about the most valuable asset we have in this growing digital economy, transforming this currency mined by tech giants into something bigger.

## Highlights
- Estimates tokens, kWh, kgCO2e, and water liters per usage
- Computes ad-impression revenue and offset coverage ratios
- Exports topic keywords and time-of-use distributions
- Simple FastAPI endpoint for JSON uploads

## Project Structure
- `CreditAnalysis370.py`: main CLI + FastAPI app (no refactor required)
- `CITATIONS.txt`: sources and formulas used in calculations
- `eco_impact_ui.html`: early UI prototype
- `analysis/`: CLI output folder (JSON + CSV)
- `examples/`: sample ChatGPT export

## Quickstart

### 1) Set up environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run CLI on your export
```bash
# Replace with your ChatGPT export path
python CreditAnalysis370.py ~/Downloads/conversations.json --outdir analysis
```
Outputs:
- `analysis/summary.json`
- `analysis/keywords.csv`
- `analysis/hourly.csv`
- `analysis/dow.csv`
- `analysis/impact.csv`

### 3) Try the FastAPI endpoint
```bash
# Start API (default: http://127.0.0.1:8000)
uvicorn CreditAnalysis370:app --reload

# Upload an export (from another terminal)
curl -X POST \
  -F "file=@examples/minimal_export.json" \
  http://127.0.0.1:8000/upload
```

## Example Data
A tiny export is included at `examples/minimal_export.json` for smoke testing.

## How It Works (Summary)
- Token estimate: characters / 4
- Energy: tokens × 0.008 Wh → kWh
- Carbon: kWh × 0.35 kgCO2e/kWh
- Water: kWh × 1.8 L/kWh
- Ads: 1 impression per user prompt, $2.50 CPM
- Offsets: $0.01/kg CO2e; water priced baseline per 1,000 gal
Details, sources, and notes are in `CITATIONS.txt`.

## Run Tests
```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

## Docker (optional)
```bash
docker build -t ecobalance .
docker run --rm -p 8000:8000 ecobalance
```

## Contributing
Contributions welcome. Please open an issue or PR. See `CITATIONS.txt` for scholarly references.

## License
MIT (see `LICENSE`).

## Acknowledgments
Inspired by attention-funded philanthropy (e.g., Ecosia). This repo explores whether ad-funded offsets can cover the environmental cost of AI usage in educational settings. This project utilized GitHub Copilot for assistance in development, see CITATIONS.txt for more details.
