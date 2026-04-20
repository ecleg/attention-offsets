## Ad Funded AI Impact Offsets

With AI being integrated into the curricula of college courses and industry, students must cast aside any reservations about its environmental impact to stay competitive. A platform that leverages the attention of students, as opposed to the dwindling savings of college students, would be able to use advertisements to pay off the damages of prompts. We need to start a discussion about the most valuable asset we have in this growing digital economy, transforming this currency mined by tech giants into something bigger.

<img width="2520" height="1631" alt="eqoC" src="https://github.com/user-attachments/assets/efedc1d5-2f72-4d5c-8b91-198198167b99" />
<img width="2521" height="1631" alt="eqoB" src="https://github.com/user-attachments/assets/dce50af2-030e-460e-9fd5-d38a3de190ee" />
<img width="2520" height="1631" alt="eqoD" src="https://github.com/user-attachments/assets/3ea2f65d-1af5-40cf-9c7e-a5ee88ea59e7" />
<img width="2521" height="1631" alt="eqoA" src="https://github.com/user-attachments/assets/916d7729-a6a9-49b3-b549-14f0d0343d87" />
<img width="2521" height="1631" alt="eqoE" src="https://github.com/user-attachments/assets/7ec02094-4dfb-43a0-bf4c-14006a2bf8dd" />


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

## Survey Snapshot (University Students)

> Purpose: Understand student AI usage, concerns, and where attention (ads) already exists.  
> Note: Response counts vary by question because some items were skipped.

### Who responded
- 85% (150) said they are currently university students (14% / 25 not students; 1% / 2 prefer not to say).

### AI usage habits
| Frequency | % | n |
|---|---:|---:|
| Weekly | 36% | 53 |
| Daily | 25% | 37 |
| Monthly | 16% | 24 |
| Never | 20% | 30 |
| Yearly | 3% | 4 |

### Tools used regularly (multi-select)
| Tool category | % | n |
|---|---:|---:|
| Chat-based assistants (e.g., ChatGPT) | 88% | 99 |
| Academic/reference tools | 36% | 40 |
| Productivity tools (summarizers/note-takers) | 17% | 19 |
| Code assistants (e.g., Copilot) | 14% | 16 |
| Image generation | 4% | 5 |

### Biggest concern about AI
| Concern | % | n |
|---|---:|---:|
| Environmental damage | 43% | 62 |
| Job losses | 20% | 28 |
| Copyright infringement | 17% | 24 |
| Other | 17% | 25 |
| No concerns | 3% | 4 |

### Where students already spend attention (multi-select)
- Instagram: 142 (~99%)
- YouTube: 128 (~89%)
- TikTok: 70 (~49%)
- Pinterest: 86 (~60%)

More detail (including question-by-question tables and marketing implications): **docs/survey-overview.md**

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
- Ads: 1 impression per user prompt, $5.00 CPM (IAB Tech vertical premium; validated against 7-user data)
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

