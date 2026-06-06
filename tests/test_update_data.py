import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import update_data


def sample_understat_rows():
    return [
        {
            'datetime': '2025-08-16 12:30:00',
            'h': {'title': 'Arsenal'},
            'a': {'title': 'Chelsea'},
            'goals': {'h': '2', 'a': '1'},
            'xG': {'h': '1.75', 'a': '0.94'},
        },
        {
            'datetime': '2025-08-16 12:30:00',
            'h': {'title': 'Arsenal'},
            'a': {'title': 'Chelsea'},
            'goals': {'h': '2', 'a': '1'},
            'xG': {'h': '1.75', 'a': '0.94'},
        },
        {
            'datetime': '2025-08-17 15:00:00',
            'h': json.dumps({'title': 'Liverpool'}),
            'a': json.dumps({'title': 'Everton'}),
            'goals': json.dumps({'h': None, 'a': '0'}),
            'xG': json.dumps({'h': '2.10', 'a': '0.50'}),
        },
    ]


def raw_frame():
    df = pd.DataFrame(sample_understat_rows())
    df['league'] = 'english_premier_league'
    df['season'] = '2025-2026'
    return df


class UpdateDataTests(unittest.TestCase):
    def test_fetch_uses_api_specific_league_name(self):
        class FakeResponse:
            text = ''

            def raise_for_status(self):
                return None

            def json(self):
                return {'dates': sample_understat_rows()[:1]}

        class FakeSession:
            def __init__(self):
                self.urls = []

            def get(self, url, headers=None, timeout=None):
                self.urls.append(url)
                return FakeResponse()

        session = FakeSession()

        update_data.fetch_understat_matches('La_liga', 2025, session=session, api_league='La liga')

        self.assertEqual(session.urls[0], 'https://understat.com/getLeagueData/La liga/2025')

    def test_extract_understat_matches_from_json_parse_payload(self):
        matches = sample_understat_rows()[:1]
        html = "<script>var datesData = JSON.parse('" + json.dumps(matches) + "')</script>"

        parsed = update_data.extract_understat_matches(html)

        self.assertEqual(parsed[0]['datetime'], '2025-08-16 12:30:00')
        self.assertEqual(parsed[0]['h']['title'], 'Arsenal')

    def test_extract_matches_from_current_api_payload(self):
        matches = sample_understat_rows()[:1]

        parsed = update_data.extract_matches_from_payload({'dates': matches, 'teams': {}, 'players': []})

        self.assertEqual(parsed, matches)

    def test_transform_and_validate_clean_schema(self):
        transformed = update_data.transform_understat_frame(raw_frame())
        clean_df, stats = update_data.validate_and_clean_data(transformed)

        self.assertEqual(list(clean_df.columns), update_data.REQUIRED_COLUMNS)
        self.assertEqual(len(clean_df), 1)
        self.assertEqual(stats['duplicate_removed'], 1)
        self.assertEqual(stats['incomplete_removed'], 1)
        self.assertEqual(clean_df.iloc[0]['home_team'], 'Arsenal')

    def test_transform_accepts_plain_team_names(self):
        df = raw_frame().head(1).copy()
        df.loc[0, 'h'] = 'Arsenal'
        df.loc[0, 'a'] = 'Chelsea'

        transformed = update_data.transform_understat_frame(df)

        self.assertEqual(transformed.iloc[0]['home_team'], 'Arsenal')
        self.assertEqual(transformed.iloc[0]['away_team'], 'Chelsea')

    def test_default_end_year_uses_current_season_start_year(self):
        self.assertEqual(update_data.default_end_year(datetime(2026, 5, 24)), 2025)
        self.assertEqual(update_data.default_end_year(datetime(2026, 8, 1)), 2026)

    def test_dry_run_does_not_write_or_train(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'football_match_data.csv'
            with patch.object(update_data, 'scrape_understat_range', return_value=raw_frame()):
                with patch.object(update_data, 'train_models', side_effect=AssertionError('training should not run')):
                    with patch.object(update_data, 'print_report'):
                        report = update_data.update_data(
                            output_path=output,
                            train=True,
                            dry_run=True,
                            pause=0,
                        )

            self.assertFalse(output.exists())
            self.assertTrue(report.dry_run)
            self.assertFalse(report.trained)
            self.assertEqual(report.final_matches, 1)

    def test_mocked_full_run_writes_backup_and_uses_training_metrics(self):
        metrics = {
            'result_model': {'accuracy': 0.5},
            'baselines': {'poisson_accuracy': 0.53},
            'score_model': {'home_mae': 0.9, 'away_mae': 0.8},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / 'football_match_data.csv'
            output.write_text('date,time,league,season,home_team,away_team,home_goals,away_goals,home_xG,away_xG\n', encoding='utf-8')

            with patch.object(update_data, 'scrape_understat_range', return_value=raw_frame()):
                with patch.object(update_data, 'train_models', return_value=metrics):
                    with patch.object(update_data, 'print_report'):
                        report = update_data.update_data(
                            output_path=output,
                            train=True,
                            dry_run=False,
                            pause=0,
                        )

            written = pd.read_csv(output)
            self.assertEqual(len(written), 1)
            self.assertTrue(Path(report.backup).exists())
            self.assertTrue(report.trained)
            self.assertEqual(report.training_metrics['baselines']['poisson_accuracy'], 0.53)


if __name__ == '__main__':
    unittest.main()
