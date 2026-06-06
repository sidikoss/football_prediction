import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


FEATURES = [
    'home_team_strength',
    'away_team_strength',
    'home_team_defense',
    'away_team_defense',
    'home_xG',
    'away_xG',
    'xG_difference',
    'home_form',
    'away_form',
    'h2h_home_win_rate',
    'h2h_away_win_rate',
    'h2h_draw_rate',
    'goal_ratio',
    'xG_ratio',
    'day_of_week',
    'month',
]


def engineer_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    df['goal_difference'] = df['home_goals'] - df['away_goals']
    df['xG_difference'] = df['home_xG'] - df['away_xG']

    window = 10
    df['home_team_strength'] = df.groupby('home_team')['home_goals'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df['away_team_strength'] = df.groupby('away_team')['away_goals'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df['home_team_defense'] = df.groupby('home_team')['away_goals'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df['away_team_defense'] = df.groupby('away_team')['home_goals'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

    df['home_form'] = df.groupby('home_team')['goal_difference'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df['away_form'] = df.groupby('away_team')['goal_difference'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )

    def get_h2h_stats(group):
        home_wins = (group['home_goals'] > group['away_goals']).sum()
        away_wins = (group['home_goals'] < group['away_goals']).sum()
        draws = (group['home_goals'] == group['away_goals']).sum()
        total_matches = len(group)
        return pd.Series(
            {
                'h2h_home_win_rate': home_wins / total_matches,
                'h2h_away_win_rate': away_wins / total_matches,
                'h2h_draw_rate': draws / total_matches,
            }
        )

    h2h_stats = df.groupby(['home_team', 'away_team']).apply(get_h2h_stats).reset_index()
    df = pd.merge(df, h2h_stats, on=['home_team', 'away_team'], how='left')

    df['result'] = np.select(
        [df['goal_difference'] > 0, df['goal_difference'] < 0, df['goal_difference'] == 0],
        ['home_win', 'away_win', 'draw'],
        default='draw',
    )
    df['goal_ratio'] = df['home_team_strength'] / (df['away_team_strength'] + 1e-5)
    df['xG_ratio'] = df['home_xG'] / (df['away_xG'] + 1e-5)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    return df.dropna(subset=['home_goals', 'away_goals'])


def classifier_pipeline(model):
    return Pipeline(
        [
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', model),
        ]
    )


def regressor_pipeline(model):
    return Pipeline(
        [
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', model),
        ]
    )


def main():
    print('Loading football_match_data.csv...')
    df = engineer_features(pd.read_csv('football_match_data.csv'))

    X = df[FEATURES]
    y_result = df['result']
    y_home_goals = df['home_goals']
    y_away_goals = df['away_goals']

    le = LabelEncoder()
    y_result_encoded = le.fit_transform(y_result)

    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_result_train = y_result_encoded[:train_size]
    y_result_test = y_result_encoded[train_size:]
    y_home_train, y_home_test = y_home_goals.iloc[:train_size], y_home_goals.iloc[train_size:]
    y_away_train, y_away_test = y_away_goals.iloc[:train_size], y_away_goals.iloc[train_size:]

    estimators = [
        (
            'RandomForest',
            classifier_pipeline(
                RandomForestClassifier(
                    n_estimators=800,
                    max_features='sqrt',
                    min_samples_leaf=2,
                    class_weight='balanced_subsample',
                    n_jobs=-1,
                    random_state=42,
                )
            ),
        ),
        (
            'ExtraTrees',
            classifier_pipeline(
                ExtraTreesClassifier(
                    n_estimators=900,
                    max_features='sqrt',
                    min_samples_leaf=2,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=43,
                )
            ),
        ),
        (
            'GradientBoosting',
            classifier_pipeline(
                GradientBoostingClassifier(
                    n_estimators=300,
                    learning_rate=0.035,
                    max_depth=3,
                    subsample=0.85,
                    random_state=44,
                )
            ),
        ),
        (
            'HistGradientBoosting',
            classifier_pipeline(
                HistGradientBoostingClassifier(
                    max_iter=550,
                    learning_rate=0.035,
                    max_leaf_nodes=31,
                    l2_regularization=0.05,
                    class_weight='balanced',
                    random_state=45,
                )
            ),
        ),
        (
            'XGBoost',
            classifier_pipeline(
                XGBClassifier(
                    n_estimators=750,
                    max_depth=5,
                    learning_rate=0.035,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.5,
                    eval_metric='mlogloss',
                    n_jobs=-1,
                    random_state=46,
                )
            ),
        ),
    ]

    print('Training ensemble result model...')
    ensemble = VotingClassifier(
        estimators=estimators,
        voting='soft',
        weights=[2, 2, 1, 2, 3],
        n_jobs=1,
    )
    ensemble.fit(X_train, y_result_train)

    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_result_test, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    print(classification_report(y_result_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_result_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

    print('Training score models...')
    home_goals_model = VotingRegressor(
        estimators=[
            (
                'RandomForest',
                regressor_pipeline(
                    RandomForestRegressor(
                        n_estimators=700,
                        max_features='sqrt',
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=52,
                    )
                ),
            ),
            (
                'ExtraTrees',
                regressor_pipeline(
                    ExtraTreesRegressor(
                        n_estimators=800,
                        max_features='sqrt',
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=53,
                    )
                ),
            ),
            (
                'HistGradientBoosting',
                regressor_pipeline(
                    HistGradientBoostingRegressor(
                        max_iter=500,
                        learning_rate=0.035,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=54,
                    )
                ),
            ),
            (
                'XGBoost',
                regressor_pipeline(
                    XGBRegressor(
                        n_estimators=700,
                        max_depth=4,
                        learning_rate=0.035,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.5,
                        objective='reg:squarederror',
                        n_jobs=-1,
                        random_state=55,
                    )
                ),
            ),
        ],
        weights=[2, 2, 2, 3],
    )
    away_goals_model = VotingRegressor(
        estimators=[
            (
                'RandomForest',
                regressor_pipeline(
                    RandomForestRegressor(
                        n_estimators=700,
                        max_features='sqrt',
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=62,
                    )
                ),
            ),
            (
                'ExtraTrees',
                regressor_pipeline(
                    ExtraTreesRegressor(
                        n_estimators=800,
                        max_features='sqrt',
                        min_samples_leaf=2,
                        n_jobs=-1,
                        random_state=63,
                    )
                ),
            ),
            (
                'HistGradientBoosting',
                regressor_pipeline(
                    HistGradientBoostingRegressor(
                        max_iter=500,
                        learning_rate=0.035,
                        max_leaf_nodes=31,
                        l2_regularization=0.05,
                        random_state=64,
                    )
                ),
            ),
            (
                'XGBoost',
                regressor_pipeline(
                    XGBRegressor(
                        n_estimators=700,
                        max_depth=4,
                        learning_rate=0.035,
                        min_child_weight=2,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.5,
                        objective='reg:squarederror',
                        n_jobs=-1,
                        random_state=65,
                    )
                ),
            ),
        ],
        weights=[2, 2, 2, 3],
    )
    home_goals_model.fit(X_train, y_home_train)
    away_goals_model.fit(X_train, y_away_train)

    home_mae = np.mean(np.abs(y_home_test - home_goals_model.predict(X_test)))
    away_mae = np.mean(np.abs(y_away_test - away_goals_model.predict(X_test)))
    print(f'Home goals MAE: {home_mae:.4f}')
    print(f'Away goals MAE: {away_mae:.4f}')

    joblib.dump(ensemble, 'football_prediction_ensemble.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    joblib.dump(home_goals_model, 'home_goals_model.joblib')
    joblib.dump(away_goals_model, 'away_goals_model.joblib')
    print('Saved football_prediction_ensemble.joblib, label_encoder.joblib, home_goals_model.joblib, away_goals_model.joblib')


if __name__ == '__main__':
    main()
