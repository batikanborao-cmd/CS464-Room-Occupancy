# Robust Room Occupancy Estimation (CS464 – Group 20)

Predicting the number of people in a room (0–3) from non-intrusive environmental sensors
(temperature, light, sound, CO₂, PIR motion), using the
[UCI Room Occupancy Estimation dataset](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) (CC BY 4.0).

## Repository layout

```
data/Occupancy_Estimation.csv   raw dataset (10,129 rows, 16 sensor features + Date/Time + label)
src/common.py                   data loading + the FIXED day-based split (used by every model/experiment)
src/eda.py                      dataset analysis -> results/eda_summary.txt, figures/*.png
src/train_rf.py                 Random Forest baseline -> results/rf_*.json/txt, figures/rf_*.png
figures/                        generated plots
results/                        generated metrics
report/                         progress report: PDF, LaTeX source (main.tex, Overleaf-ready), build_report.py
```

## How to run

```bash
pip install -r requirements.txt
cd src
python eda.py
python train_rf.py
```

## Evaluation protocol (follows TA feedback)

- **Day-based split** to avoid leakage between consecutive 30 s readings. Validation days are separate from test days.
- **The same split is used for all models and experiments** (defined once in `src/common.py`).
- **Macro-F1 is the main metric** (81% of samples are "empty room"); accuracy is reported as an extra score.
- Hyperparameter tuning uses **validation only**; final comparison is on the **same test set**.

Only 3 of the 7 days contain people (12/22, 12/23, 01/10), and class 1 appears only on 12/22 and 12/23.
Each split therefore gets exactly one occupied day:

| Split | Days | 0 | 1 | 2 | 3 | Total |
|---|---|---|---|---|---|---|
| Train | 2017-12-22, 12-24, 12-25, 12-26 | 4480 | 205 | 257 | 363 | 5305 (52%) |
| Val | 2018-01-10, 01-11 | 1751 | 0 | 93 | 201 | 2045 (20%) |
| Test | 2017-12-23 | 1997 | 254 | 398 | 130 | 2779 (27%) |

The validation set has no class-1 samples, so validation macro-F1 is averaged over the classes present (0, 2, 3).

## Progress-report results: Random Forest (sklearn defaults, seed 42)

| Evaluation | Macro-F1 | Accuracy |
|---|---|---|
| Random 80/20 split (leaky, reference only) | 0.990 | 0.997 |
| Day split – validation (01-10/11, classes 0/2/3) | 0.429 | 0.868 |
| Day split – test (12-23) | **0.906** | 0.945 |

Per-class test F1: 0 → 0.971, 1 → 0.972, 2 → 0.780, 3 → 0.901. Most errors: true class 2 predicted as 0 (111 samples).

Observations:
- The random split hides the difficulty of the task: consecutive readings are near-duplicates.
- Light sensors (S1, S2, S3) and CO₂ slope dominate the feature importance; PIR sensors are almost unused.
- Validation day 2018-01-10 shows a strong **distribution shift**: with 3 occupants, mean S1/S2 light is ~18–26 lux
  vs. ~150–240 lux in December, so the model (which relies on light) predicts most class-3 samples as empty.
  This directly motivates the planned sensor ablation, noise and missing-sensor experiments.

## Next steps (final report)

Logistic Regression, XGBoost and MLP on the same split; hyperparameter tuning on validation; sensor ablation,
noise robustness, missing-sensor simulation, class-imbalance handling, efficiency analysis.
