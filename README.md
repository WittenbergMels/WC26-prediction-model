# WC2026 Group Stage Prediction Model

Multi-factor ML-inspired model for predicting the 2026 FIFA World Cup group stage.

## How it works

Six factors combine to compute **expected goals (xG)** per team per match, fed into a Poisson score predictor:

| Factor | How it's used |
|---|---|
| **ELO rating** | Core strength — `(elo_a/elo_b)^1.5 × avg_goals` |
| **Recent form** | Win-rate above/below 55% adjusts xG by ±0.15 |
| **Squad height** | Taller teams get a set-piece bonus (±0.1 xG per 5cm vs 181.5cm avg) |
| **Age × heat penalty** | Older squads playing hotter than their home climate lose up to 0.2 xG |
| **Goalkeeper quality** | Rated 6–10; elite GK reduces opponent xG by up to 8% |
| **Head-to-head** | Historical rivalries adjust xG (e.g. Mexico +0.12 vs USA) |
| **Underdog motivation** | Teams with <20% win prob get an extra 0.08 xG boost |

## Usage

```bash
python3 wc2026_predictor.py
```

Outputs `output.csv` with all 72 group stage match predictions in the required format.

## Files

- `wc2026_predictor.py` — full prediction model
- `output.csv` — predicted scores for all 72 group stage matches
- `output_template 1.csv` — original blank template
