# Football Match Prediction

This project aims to predict the outcome and scores of football matches using machine learning models. The application utilizes an ensemble of multiple classifiers and is capable of predicting both match outcomes and exact scores for home and away teams.

## Features

- Predicts match outcomes (home win, away win, draw)
- Predicts exact scores for home and away teams
- Displays probabilities of each possible outcome
- Provides head-to-head history between selected teams
- Shows recent match history for both home and away teams

## Model Performance

Use `MODEL_EVALUATION.md` as the source of truth for current metrics. The
current evaluation uses leakage-free, time-based validation: rolling team stats,
form, and head-to-head values are calculated only from matches that happened
before the match being evaluated.

Older 70%+ metrics in historical notes came from validation setups that used
future or same-match information. They are useful as development history, but
not as production-quality accuracy numbers.

Current production selection is based on the best honest time-split 1X2
accuracy between the ML ensemble and the historical Poisson baseline.

## Model Components

### Ensemble Model

The ensemble model combines multiple machine learning algorithms:

- Random Forest Classifier
- XGBoost Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM) Classifier

### Voting Mechanism

- Uses "soft" voting: averages the predicted probabilities from each model and selects the class with the highest average probability as the final prediction.

### Feature Processing

- **Imputation**: Filling in missing values using the mean of the column.
- **Scaling**: Standardizing the features so they are on the same scale.
- **Feature Selection**: Using a Random Forest to select the most important features.

### Key Features

- Team strengths (offensive and defensive)
- Expected goals (xG) for home and away teams
- Head-to-head statistics
- Recent form of both teams

## Data Updates and Training

The primary update command rebuilds the full Understat dataset, validates it,
backs up the existing CSV, writes `football_match_data.csv`, and optionally
re-trains the honest model:

```powershell
.\venv\Scripts\python.exe update_data.py --train
```

Useful options:

- `--dry-run`: fetch and validate without writing files or training
- `--skip-train`: update the CSV without retraining models
- `--start-year` / `--end-year`: override the season-start year range
- `--output`: write to a different CSV path

The update pipeline removes incomplete matches, de-duplicates by match identity,
and fails before replacing the existing CSV if Understat is unavailable or the
schema is invalid. Backups are written under `data_backups/`.

## Flask API

The Flask API provides endpoints for making predictions and fetching team lists:

### Endpoints

- `/`: Renders the homepage.
- `/predict`: Accepts POST requests with `home_team` and `away_team` to predict match outcomes and scores.
- `/teams`: Returns a list of unique teams.

### Usage

1. Run the Flask app:

   Use the command to run `app.py`.

2. Access the homepage at `http://localhost:5000`.

### Live Match Sync

The app can show live matches from Football-Data.org and send a live fixture
directly into the prediction form.

1. Create a Football-Data.org API token.
2. Set the token before starting Flask:

   ```powershell
   $env:FOOTBALL_DATA_API_TOKEN="your_api_token"
   .\venv\Scripts\python.exe app.py
   ```

3. Open `http://localhost:5000` and use the **Live Matches** panel.

Optional settings:

- `FOOTBALL_DATA_API_BASE_URL`: defaults to `https://api.football-data.org/v4`
- `LIVE_MATCH_CACHE_SECONDS`: defaults to `60`

If `FOOTBALL_DATA_API_TOKEN` is not set, the app keeps working in manual
prediction mode and displays setup instructions in the live panel.

## HTML Interface

The HTML interface allows users to select teams and view predictions. It includes:

- Dropdowns for selecting home and away teams
- Loading spinner while predictions are being processed
- Display of predicted outcome and score
- Bar chart showing probabilities of each possible outcome
- Tables displaying head-to-head history and recent match history for both teams

## Installation

1. Clone the repository from GitHub.
2. Install the required packages listed in `requirements.txt`.
3. Update data and train with `update_data.py --train`.
4. Run the Flask app using `app.py`.

## Model Storage

The model files are too large to be stored on GitHub. They are available on Kaggle:

- [Football Prediction Ensemble Model](https://www.kaggle.com/models/stevemcs/football_prediction_ensemble/)

## Understanding xG (Expected Goals) in Football Predictions

### What is xG?

xG, or Expected Goals, is a statistical measure in football that represents the quality of scoring chances created by a team in a match. It's based on historical data of similar chances and how often they resulted in goals.

### Simple explanation:

Imagine if every shot in football had a "difficulty rating" from 0 to 1:

- 0 means it's impossible to score
- 1 means it's a certain goal

xG is the sum of these "difficulty ratings" for all shots a team takes in a match.

### Examples:

1. A penalty kick might have an xG of 0.76 (76% chance of scoring)
2. A shot from the halfway line might have an xG of 0.01 (1% chance of scoring)
3. A tap-in from 1 yard out might have an xG of 0.96 (96% chance of scoring)

If a team takes these three shots in a match, their total xG would be:
0.76 + 0.01 + 0.96 = 1.73

This means, based on their chances, they would be expected to score about 1.73 goals on average.

### Why use xG in predictions?

1. It's more accurate than just looking at goals scored
2. It accounts for the quality of chances, not just the quantity
3. It can indicate if a team is performing better or worse than their actual goal tally suggests

### How to use xG in our prediction model:

1. Look up recent xG statistics for both teams
2. Input these values into our prediction form
3. The model uses this information, along with other factors, to predict the match outcome

### Where to find xG data:

- Websites like Understat, FBref, or WhoScored often provide xG data for matches
- You can calculate an average xG for recent matches if you don't have data for a specific upcoming match

Remember, xG is just one factor in predicting match outcomes, but it's a valuable one that gives insight into team performance beyond just goals scored.
