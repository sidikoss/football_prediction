import json
import math
from datetime import datetime

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from honest_features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, build_training_frame


def make_preprocessor():
    numeric_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ('numeric', numeric_pipeline, NUMERIC_FEATURES),
            ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0,
    )


def make_result_model():
    return Pipeline(
        [
            ('preprocess', make_preprocessor()),
            (
                'model',
                VotingClassifier(
                    estimators=[
                        (
                            'hist_gb',
                            HistGradientBoostingClassifier(
                                max_iter=350,
                                learning_rate=0.035,
                                max_leaf_nodes=31,
                                l2_regularization=0.1,
                                class_weight='balanced',
                                random_state=101,
                            ),
                        ),
                        (
                            'random_forest',
                            RandomForestClassifier(
                                n_estimators=450,
                                max_features='sqrt',
                                min_samples_leaf=6,
                                class_weight='balanced_subsample',
                                n_jobs=-1,
                                random_state=102,
                            ),
                        ),
                        (
                            'extra_trees',
                            ExtraTreesClassifier(
                                n_estimators=550,
                                max_features='sqrt',
                                min_samples_leaf=6,
                                class_weight='balanced',
                                n_jobs=-1,
                                random_state=103,
                            ),
                        ),
                        (
                            'logistic',
                            LogisticRegression(
                                C=0.8,
                                max_iter=2000,
                                class_weight='balanced',
                                random_state=104,
                            ),
                        ),
                    ],
                    voting='soft',
                    weights=[3, 2, 2, 1],
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_score_model(random_state):
    return Pipeline(
        [
            ('preprocess', make_preprocessor()),
            (
                'model',
                VotingRegressor(
                    estimators=[
                        (
                            'hist_gb',
                            HistGradientBoostingRegressor(
                                max_iter=350,
                                learning_rate=0.035,
                                max_leaf_nodes=31,
                                l2_regularization=0.1,
                                random_state=random_state,
                            ),
                        ),
                        (
                            'random_forest',
                            RandomForestRegressor(
                                n_estimators=350,
                                max_features='sqrt',
                                min_samples_leaf=6,
                                n_jobs=-1,
                                random_state=random_state + 1,
                            ),
                        ),
                        (
                            'extra_trees',
                            ExtraTreesRegressor(
                                n_estimators=450,
                                max_features='sqrt',
                                min_samples_leaf=6,
                                n_jobs=-1,
                                random_state=random_state + 2,
                            ),
                        ),
                    ],
                    weights=[3, 2, 1],
                ),
            ),
        ]
    )


def poisson_pmf(goal_count, expected_goals):
    expected_goals = max(float(expected_goals), 0.05)
    return math.exp(-expected_goals) * (expected_goals ** goal_count) / math.factorial(goal_count)


def poisson_result_probabilities(home_expected, away_expected, max_goals=10):
    home_probs = [poisson_pmf(i, home_expected) for i in range(max_goals + 1)]
    away_probs = [poisson_pmf(i, away_expected) for i in range(max_goals + 1)]
    home_win = 0
    draw = 0
    away_win = 0
    for home_goals, home_prob in enumerate(home_probs):
        for away_goals, away_prob in enumerate(away_probs):
            probability = home_prob * away_prob
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability
    total = home_win + draw + away_win
    return np.array([away_win / total, draw / total, home_win / total])


def poisson_expected_goals(row, fallback_home, fallback_away):
    if 'poisson_home_expected_goals' in row and 'poisson_away_expected_goals' in row:
        home_expected = row['poisson_home_expected_goals']
        away_expected = row['poisson_away_expected_goals']
        if not np.isnan(home_expected) and not np.isnan(away_expected):
            return max(float(home_expected), 0.05), max(float(away_expected), 0.05)

    home_candidates = [
        row['home_avg_xg_for'],
        row['away_avg_xg_against'],
        row['home_home_xg_for'],
        row['away_away_xg_against'],
    ]
    away_candidates = [
        row['away_avg_xg_for'],
        row['home_avg_xg_against'],
        row['away_away_xg_for'],
        row['home_home_xg_against'],
    ]
    home_expected = np.nanmean(home_candidates)
    away_expected = np.nanmean(away_candidates)
    if np.isnan(home_expected):
        home_expected = fallback_home
    if np.isnan(away_expected):
        away_expected = fallback_away
    return max(float(home_expected), 0.05), max(float(away_expected), 0.05)


def brier_multiclass(y_true, probabilities, class_count):
    observed = np.eye(class_count)[y_true]
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def expected_calibration_error(y_true, probabilities, bins=10):
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == y_true
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        mask = (confidence > lower) & (confidence <= upper)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def write_report(metrics):
    report = f"""# Honest Model Evaluation

Generated: {metrics['generated_at']}

This report uses only pre-match features. Rolling team stats, form and head-to-head values are calculated from matches that happened before the match being evaluated.

## Dataset

- Matches used after minimum history filter: {metrics['dataset']['matches_used']}
- Training matches: {metrics['dataset']['train_matches']}
- Test matches: {metrics['dataset']['test_matches']}
- Test date range: {metrics['dataset']['test_start']} to {metrics['dataset']['test_end']}

## Result Prediction

- Honest ML accuracy: {metrics['result_model']['accuracy']:.4f}
- Honest ML log loss: {metrics['result_model']['log_loss']:.4f}
- Honest ML multiclass Brier: {metrics['result_model']['brier']:.4f}
- Honest ML ECE: {metrics['result_model']['ece']:.4f}
- Majority-class baseline accuracy: {metrics['baselines']['majority_accuracy']:.4f}
- Historical Poisson baseline accuracy: {metrics['baselines']['poisson_accuracy']:.4f}
- Production result method: {metrics['production']['result_method']}

## Score Prediction

- Home goals MAE: {metrics['score_model']['home_mae']:.4f}
- Away goals MAE: {metrics['score_model']['away_mae']:.4f}
- Exact rounded score accuracy: {metrics['score_model']['exact_score_accuracy']:.4f}
- Over 2.5 accuracy from score model: {metrics['score_model']['over_2_5_accuracy']:.4f}
- BTTS accuracy from score model: {metrics['score_model']['btts_accuracy']:.4f}

## Important Notes

- This number is lower than leaky validation by design.
- No current-match xG, current-match goals, or future head-to-head data are used.
- Probabilities should be read as model estimates, not guarantees.
"""
    with open('MODEL_EVALUATION.md', 'w', encoding='utf-8') as file:
        file.write(report)


def main():
    print('Building leakage-free pre-match features...')
    raw_df = pd.read_csv('football_match_data.csv')
    frame = build_training_frame(raw_df, min_prior_matches=5)
    frame = frame.sort_values(['date', 'home_team', 'away_team']).reset_index(drop=True)

    X = frame[FEATURE_COLUMNS]
    y_result = frame['result']
    y_home_goals = frame['home_goals']
    y_away_goals = frame['away_goals']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_result)

    train_size = int(len(frame) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y_encoded[:train_size], y_encoded[train_size:]
    y_home_train, y_home_test = y_home_goals.iloc[:train_size], y_home_goals.iloc[train_size:]
    y_away_train, y_away_test = y_away_goals.iloc[:train_size], y_away_goals.iloc[train_size:]

    print(f'Train matches: {len(X_train)}')
    print(f'Test matches: {len(X_test)}')
    print(f'Test range: {frame.iloc[train_size]["date"].date()} to {frame.iloc[-1]["date"].date()}')

    result_model = make_result_model()
    print('Training honest result model...')
    result_model.fit(X_train, y_train)

    probabilities = result_model.predict_proba(X_test)
    predictions = probabilities.argmax(axis=1)
    accuracy = accuracy_score(y_test, predictions)
    model_log_loss = log_loss(y_test, probabilities, labels=list(range(len(label_encoder.classes_))))
    model_brier = brier_multiclass(y_test, probabilities, len(label_encoder.classes_))
    model_ece = expected_calibration_error(y_test, probabilities)

    majority_class = pd.Series(y_train).mode().iloc[0]
    majority_accuracy = accuracy_score(y_test, np.full_like(y_test, majority_class))

    fallback_home = float(y_home_train.mean())
    fallback_away = float(y_away_train.mean())
    poisson_probabilities = np.vstack([
        poisson_result_probabilities(*poisson_expected_goals(row, fallback_home, fallback_away))
        for _, row in X_test.iterrows()
    ])
    poisson_predictions = poisson_probabilities.argmax(axis=1)
    poisson_accuracy = accuracy_score(y_test, poisson_predictions)

    print(f'Honest accuracy: {accuracy:.4f}')
    print(f'Honest log loss: {model_log_loss:.4f}')
    print(f'Poisson baseline accuracy: {poisson_accuracy:.4f}')
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Honest Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    print('Training honest score models...')
    home_goals_model = make_score_model(201)
    away_goals_model = make_score_model(301)
    home_goals_model.fit(X_train, y_home_train)
    away_goals_model.fit(X_train, y_away_train)

    home_predictions = np.clip(home_goals_model.predict(X_test), 0, None)
    away_predictions = np.clip(away_goals_model.predict(X_test), 0, None)
    home_mae = float(np.mean(np.abs(y_home_test - home_predictions)))
    away_mae = float(np.mean(np.abs(y_away_test - away_predictions)))
    rounded_home = np.rint(home_predictions).astype(int)
    rounded_away = np.rint(away_predictions).astype(int)
    exact_score_accuracy = float(np.mean((rounded_home == y_home_test.to_numpy()) & (rounded_away == y_away_test.to_numpy())))
    over_2_5_accuracy = float(np.mean(((home_predictions + away_predictions) > 2.5) == ((y_home_test + y_away_test).to_numpy() > 2.5)))
    btts_accuracy = float(np.mean(((home_predictions > 0.5) & (away_predictions > 0.5)) == ((y_home_test.to_numpy() > 0) & (y_away_test.to_numpy() > 0))))

    print(f'Home goals MAE: {home_mae:.4f}')
    print(f'Away goals MAE: {away_mae:.4f}')
    print(f'Exact rounded score accuracy: {exact_score_accuracy:.4f}')

    metrics = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'method': 'leakage_free_time_split',
        'dataset': {
            'matches_used': int(len(frame)),
            'train_matches': int(len(X_train)),
            'test_matches': int(len(X_test)),
            'test_start': str(frame.iloc[train_size]['date'].date()),
            'test_end': str(frame.iloc[-1]['date'].date()),
        },
        'features': {
            'categorical': CATEGORICAL_FEATURES,
            'numeric': NUMERIC_FEATURES,
        },
        'result_model': {
            'accuracy': float(accuracy),
            'log_loss': float(model_log_loss),
            'brier': model_brier,
            'ece': model_ece,
            'classes': label_encoder.classes_.tolist(),
        },
        'baselines': {
            'majority_accuracy': float(majority_accuracy),
            'poisson_accuracy': float(poisson_accuracy),
        },
        'production': {
            'result_method': 'historical_poisson' if poisson_accuracy >= accuracy else 'ml_ensemble',
            'reason': 'Selected by highest honest time-split 1X2 accuracy.',
        },
        'score_model': {
            'home_mae': home_mae,
            'away_mae': away_mae,
            'exact_score_accuracy': exact_score_accuracy,
            'over_2_5_accuracy': over_2_5_accuracy,
            'btts_accuracy': btts_accuracy,
        },
    }

    with open('model_evaluation.json', 'w', encoding='utf-8') as file:
        json.dump(metrics, file, indent=2)
    write_report(metrics)

    joblib.dump(result_model, 'football_prediction_ensemble.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')
    joblib.dump(home_goals_model, 'home_goals_model.joblib')
    joblib.dump(away_goals_model, 'away_goals_model.joblib')
    joblib.dump(
        {
            'feature_columns': FEATURE_COLUMNS,
            'categorical_features': CATEGORICAL_FEATURES,
            'numeric_features': NUMERIC_FEATURES,
            'metrics': metrics,
        },
        'model_metadata.joblib',
    )
    print('Saved honest models and evaluation report.')


if __name__ == '__main__':
    main()
