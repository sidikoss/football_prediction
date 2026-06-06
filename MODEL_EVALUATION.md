# Honest Model Evaluation

Generated: 2026-05-24T11:08:54

This report uses only pre-match features. Rolling team stats, form and head-to-head values are calculated from matches that happened before the match being evaluated.

## Dataset

- Matches used after minimum history filter: 19282
- Training matches: 15425
- Test matches: 3857
- Test date range: 2023-04-21 to 2024-06-02

## Result Prediction

- Honest ML accuracy: 0.5136
- Honest ML log loss: 0.9898
- Honest ML multiclass Brier: 0.5901
- Honest ML ECE: 0.0517
- Majority-class baseline accuracy: 0.4402
- Historical Poisson baseline accuracy: 0.5312
- Production result method: historical_poisson

## Score Prediction

- Home goals MAE: 0.9619
- Away goals MAE: 0.8662
- Exact rounded score accuracy: 0.1190
- Over 2.5 accuracy from score model: 0.5655
- BTTS accuracy from score model: 0.5585

## Important Notes

- This number is lower than leaky validation by design.
- No current-match xG, current-match goals, or future head-to-head data are used.
- Probabilities should be read as model estimates, not guarantees.
