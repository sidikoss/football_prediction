from collections import defaultdict, deque
from datetime import datetime
import math

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = ['league']

NUMERIC_FEATURES = [
    'home_matches_played',
    'away_matches_played',
    'home_avg_goals_for',
    'home_avg_goals_against',
    'away_avg_goals_for',
    'away_avg_goals_against',
    'home_avg_xg_for',
    'home_avg_xg_against',
    'away_avg_xg_for',
    'away_avg_xg_against',
    'home_points_per_match',
    'away_points_per_match',
    'home_recent_points',
    'away_recent_points',
    'home_recent_goal_diff',
    'away_recent_goal_diff',
    'home_recent_xg_diff',
    'away_recent_xg_diff',
    'home_home_goals_for',
    'home_home_goals_against',
    'away_away_goals_for',
    'away_away_goals_against',
    'home_home_xg_for',
    'home_home_xg_against',
    'away_away_xg_for',
    'away_away_xg_against',
    'team_strength_diff',
    'team_defense_diff',
    'xg_attack_diff',
    'xg_defense_diff',
    'form_points_diff',
    'form_goal_diff',
    'venue_attack_diff',
    'venue_defense_diff',
    'poisson_home_expected_goals',
    'poisson_away_expected_goals',
    'poisson_home_win_probability',
    'poisson_draw_probability',
    'poisson_away_win_probability',
    'poisson_over_2_5_probability',
    'poisson_btts_probability',
    'home_rest_days',
    'away_rest_days',
    'h2h_matches',
    'h2h_home_team_win_rate',
    'h2h_draw_rate',
    'h2h_away_team_win_rate',
    'h2h_home_goals_for',
    'h2h_away_goals_for',
    'day_of_week',
    'month',
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _new_team_state():
    return {
        'games': 0,
        'goals_for': 0.0,
        'goals_against': 0.0,
        'xg_for': 0.0,
        'xg_against': 0.0,
        'points': 0.0,
        'home_games': 0,
        'home_goals_for': 0.0,
        'home_goals_against': 0.0,
        'home_xg_for': 0.0,
        'home_xg_against': 0.0,
        'away_games': 0,
        'away_goals_for': 0.0,
        'away_goals_against': 0.0,
        'away_xg_for': 0.0,
        'away_xg_against': 0.0,
        'recent_points': deque(maxlen=5),
        'recent_goal_diff': deque(maxlen=5),
        'recent_xg_diff': deque(maxlen=5),
        'last_date': None,
        'league_counts': defaultdict(int),
    }


def _safe_divide(numerator, denominator):
    return np.nan if denominator == 0 else numerator / denominator


def _safe_mean(values):
    return np.nan if not values else float(np.mean(values))


def _nanmean(values):
    clean_values = [value for value in values if not pd.isna(value)]
    return np.nan if not clean_values else float(np.mean(clean_values))


def _pair_key(home_team, away_team):
    return tuple(sorted((home_team, away_team)))


def _points_for(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _poisson_pmf(goal_count, expected_goals):
    expected_goals = max(float(expected_goals), 0.05)
    return math.exp(-expected_goals) * (expected_goals ** goal_count) / math.factorial(goal_count)


def _poisson_features(home_expected, away_expected, max_goals=10):
    if np.isnan(home_expected) or np.isnan(away_expected):
        return {
            'poisson_home_expected_goals': np.nan,
            'poisson_away_expected_goals': np.nan,
            'poisson_home_win_probability': np.nan,
            'poisson_draw_probability': np.nan,
            'poisson_away_win_probability': np.nan,
            'poisson_over_2_5_probability': np.nan,
            'poisson_btts_probability': np.nan,
        }

    home_expected = max(float(home_expected), 0.05)
    away_expected = max(float(away_expected), 0.05)
    home_probs = [_poisson_pmf(i, home_expected) for i in range(max_goals + 1)]
    away_probs = [_poisson_pmf(i, away_expected) for i in range(max_goals + 1)]

    home_win = 0
    draw = 0
    away_win = 0
    over_2_5 = 0

    for home_goals, home_prob in enumerate(home_probs):
        for away_goals, away_prob in enumerate(away_probs):
            probability = home_prob * away_prob
            if home_goals > away_goals:
                home_win += probability
            elif home_goals == away_goals:
                draw += probability
            else:
                away_win += probability
            if home_goals + away_goals > 2.5:
                over_2_5 += probability

    total = home_win + draw + away_win
    btts = 1 - math.exp(-home_expected) - math.exp(-away_expected) + math.exp(-(home_expected + away_expected))
    return {
        'poisson_home_expected_goals': home_expected,
        'poisson_away_expected_goals': away_expected,
        'poisson_home_win_probability': home_win / total,
        'poisson_draw_probability': draw / total,
        'poisson_away_win_probability': away_win / total,
        'poisson_over_2_5_probability': over_2_5 / total,
        'poisson_btts_probability': btts,
    }


def _infer_league(home_state, away_state, fallback='unknown'):
    counts = defaultdict(int)
    for league, count in home_state['league_counts'].items():
        counts[league] += count
    for league, count in away_state['league_counts'].items():
        counts[league] += count
    if not counts:
        return fallback
    return max(counts.items(), key=lambda item: item[1])[0]


def _h2h_features(pair_history, home_team, away_team):
    if not pair_history:
        return {
            'h2h_matches': 0,
            'h2h_home_team_win_rate': np.nan,
            'h2h_draw_rate': np.nan,
            'h2h_away_team_win_rate': np.nan,
            'h2h_home_goals_for': np.nan,
            'h2h_away_goals_for': np.nan,
        }

    home_wins = 0
    draws = 0
    away_wins = 0
    home_goals = []
    away_goals = []

    for match in pair_history:
        if match['home_team'] == home_team:
            current_home_goals = match['home_goals']
            current_away_goals = match['away_goals']
        else:
            current_home_goals = match['away_goals']
            current_away_goals = match['home_goals']

        home_goals.append(current_home_goals)
        away_goals.append(current_away_goals)
        if current_home_goals > current_away_goals:
            home_wins += 1
        elif current_home_goals == current_away_goals:
            draws += 1
        else:
            away_wins += 1

    total = len(pair_history)
    return {
        'h2h_matches': total,
        'h2h_home_team_win_rate': home_wins / total,
        'h2h_draw_rate': draws / total,
        'h2h_away_team_win_rate': away_wins / total,
        'h2h_home_goals_for': float(np.mean(home_goals)),
        'h2h_away_goals_for': float(np.mean(away_goals)),
    }


def _days_since(last_date, current_date):
    if last_date is None:
        return np.nan
    return max((current_date - last_date).days, 0)


def _build_feature_row(context, home_team, away_team, match_date, league=None):
    home_state = context['team_stats'][home_team]
    away_state = context['team_stats'][away_team]
    pair_history = context['pair_history'][_pair_key(home_team, away_team)]

    home_avg_goals_for = _safe_divide(home_state['goals_for'], home_state['games'])
    home_avg_goals_against = _safe_divide(home_state['goals_against'], home_state['games'])
    away_avg_goals_for = _safe_divide(away_state['goals_for'], away_state['games'])
    away_avg_goals_against = _safe_divide(away_state['goals_against'], away_state['games'])
    home_avg_xg_for = _safe_divide(home_state['xg_for'], home_state['games'])
    home_avg_xg_against = _safe_divide(home_state['xg_against'], home_state['games'])
    away_avg_xg_for = _safe_divide(away_state['xg_for'], away_state['games'])
    away_avg_xg_against = _safe_divide(away_state['xg_against'], away_state['games'])

    home_home_goals_for = _safe_divide(home_state['home_goals_for'], home_state['home_games'])
    home_home_goals_against = _safe_divide(home_state['home_goals_against'], home_state['home_games'])
    away_away_goals_for = _safe_divide(away_state['away_goals_for'], away_state['away_games'])
    away_away_goals_against = _safe_divide(away_state['away_goals_against'], away_state['away_games'])
    home_home_xg_for = _safe_divide(home_state['home_xg_for'], home_state['home_games'])
    home_home_xg_against = _safe_divide(home_state['home_xg_against'], home_state['home_games'])
    away_away_xg_for = _safe_divide(away_state['away_xg_for'], away_state['away_games'])
    away_away_xg_against = _safe_divide(away_state['away_xg_against'], away_state['away_games'])

    home_recent_points = _safe_mean(home_state['recent_points'])
    away_recent_points = _safe_mean(away_state['recent_points'])
    home_recent_goal_diff = _safe_mean(home_state['recent_goal_diff'])
    away_recent_goal_diff = _safe_mean(away_state['recent_goal_diff'])
    home_recent_xg_diff = _safe_mean(home_state['recent_xg_diff'])
    away_recent_xg_diff = _safe_mean(away_state['recent_xg_diff'])

    row = {
        'league': league or _infer_league(home_state, away_state),
        'home_matches_played': home_state['games'],
        'away_matches_played': away_state['games'],
        'home_avg_goals_for': home_avg_goals_for,
        'home_avg_goals_against': home_avg_goals_against,
        'away_avg_goals_for': away_avg_goals_for,
        'away_avg_goals_against': away_avg_goals_against,
        'home_avg_xg_for': home_avg_xg_for,
        'home_avg_xg_against': home_avg_xg_against,
        'away_avg_xg_for': away_avg_xg_for,
        'away_avg_xg_against': away_avg_xg_against,
        'home_points_per_match': _safe_divide(home_state['points'], home_state['games']),
        'away_points_per_match': _safe_divide(away_state['points'], away_state['games']),
        'home_recent_points': home_recent_points,
        'away_recent_points': away_recent_points,
        'home_recent_goal_diff': home_recent_goal_diff,
        'away_recent_goal_diff': away_recent_goal_diff,
        'home_recent_xg_diff': home_recent_xg_diff,
        'away_recent_xg_diff': away_recent_xg_diff,
        'home_home_goals_for': home_home_goals_for,
        'home_home_goals_against': home_home_goals_against,
        'away_away_goals_for': away_away_goals_for,
        'away_away_goals_against': away_away_goals_against,
        'home_home_xg_for': home_home_xg_for,
        'home_home_xg_against': home_home_xg_against,
        'away_away_xg_for': away_away_xg_for,
        'away_away_xg_against': away_away_xg_against,
        'team_strength_diff': home_avg_goals_for - away_avg_goals_for,
        'team_defense_diff': away_avg_goals_against - home_avg_goals_against,
        'xg_attack_diff': home_avg_xg_for - away_avg_xg_for,
        'xg_defense_diff': away_avg_xg_against - home_avg_xg_against,
        'form_points_diff': home_recent_points - away_recent_points,
        'form_goal_diff': home_recent_goal_diff - away_recent_goal_diff,
        'venue_attack_diff': home_home_goals_for - away_away_goals_for,
        'venue_defense_diff': away_away_goals_against - home_home_goals_against,
        'home_rest_days': _days_since(home_state['last_date'], match_date),
        'away_rest_days': _days_since(away_state['last_date'], match_date),
        'day_of_week': match_date.weekday(),
        'month': match_date.month,
    }
    row.update(_poisson_features(
        _nanmean([home_avg_xg_for, away_avg_xg_against, home_home_xg_for, away_away_xg_against]),
        _nanmean([away_avg_xg_for, home_avg_xg_against, away_away_xg_for, home_home_xg_against]),
    ))
    row.update(_h2h_features(pair_history, home_team, away_team))
    return row


def _update_team_state(state, date, league, goals_for, goals_against, xg_for, xg_against, venue):
    points = _points_for(goals_for, goals_against)
    state['games'] += 1
    state['goals_for'] += goals_for
    state['goals_against'] += goals_against
    state['xg_for'] += xg_for
    state['xg_against'] += xg_against
    state['points'] += points
    state['recent_points'].append(points)
    state['recent_goal_diff'].append(goals_for - goals_against)
    state['recent_xg_diff'].append(xg_for - xg_against)
    state['last_date'] = date
    state['league_counts'][league] += 1

    prefix = 'home' if venue == 'home' else 'away'
    state[f'{prefix}_games'] += 1
    state[f'{prefix}_goals_for'] += goals_for
    state[f'{prefix}_goals_against'] += goals_against
    state[f'{prefix}_xg_for'] += xg_for
    state[f'{prefix}_xg_against'] += xg_against


def _update_context(context, match):
    date = match['date']
    league = match['league']
    home_team = match['home_team']
    away_team = match['away_team']
    home_goals = float(match['home_goals'])
    away_goals = float(match['away_goals'])
    home_xg = float(match['home_xG'])
    away_xg = float(match['away_xG'])

    _update_team_state(context['team_stats'][home_team], date, league, home_goals, away_goals, home_xg, away_xg, 'home')
    _update_team_state(context['team_stats'][away_team], date, league, away_goals, home_goals, away_xg, home_xg, 'away')
    context['pair_history'][_pair_key(home_team, away_team)].append({
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': home_goals,
        'away_goals': away_goals,
    })


def empty_context():
    return {
        'team_stats': defaultdict(_new_team_state),
        'pair_history': defaultdict(list),
    }


def prepare_match_data(raw_df):
    df = raw_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['home_goals', 'away_goals', 'home_xG', 'away_xG'])
    return df.sort_values(['date', 'time', 'home_team', 'away_team']).reset_index(drop=True)


def build_training_frame(raw_df, min_prior_matches=5):
    df = prepare_match_data(raw_df)
    context = empty_context()
    rows = []

    for _, match in df.iterrows():
        row = _build_feature_row(context, match['home_team'], match['away_team'], match['date'], match['league'])
        row.update({
            'date': match['date'],
            'season': match['season'],
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_goals': float(match['home_goals']),
            'away_goals': float(match['away_goals']),
            'result': 'home_win' if match['home_goals'] > match['away_goals'] else 'away_win' if match['home_goals'] < match['away_goals'] else 'draw',
        })
        rows.append(row)
        _update_context(context, match)

    feature_df = pd.DataFrame(rows)
    return feature_df[
        (feature_df['home_matches_played'] >= min_prior_matches)
        & (feature_df['away_matches_played'] >= min_prior_matches)
    ].reset_index(drop=True)


def build_feature_context(raw_df):
    df = prepare_match_data(raw_df)
    context = empty_context()
    for _, match in df.iterrows():
        _update_context(context, match)
    return context


def build_prediction_frame(context, home_team, away_team, match_date=None, league=None):
    if match_date is None:
        match_date = datetime.now()
    if not isinstance(match_date, pd.Timestamp):
        match_date = pd.Timestamp(match_date)
    row = _build_feature_row(context, home_team, away_team, match_date, league)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
