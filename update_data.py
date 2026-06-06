import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


LEAGUES = {
    'EPL': {'api': 'EPL', 'name': 'english_premier_league'},
    'La_liga': {'api': 'La liga', 'name': 'spanish_la_liga'},
    'Bundesliga': {'api': 'Bundesliga', 'name': 'german_bundesliga'},
    'Serie_A': {'api': 'Serie A', 'name': 'italian_serie_a'},
    'Ligue_1': {'api': 'Ligue 1', 'name': 'french_ligue_1'},
}

REQUIRED_COLUMNS = [
    'date',
    'time',
    'league',
    'season',
    'home_team',
    'away_team',
    'home_goals',
    'away_goals',
    'home_xG',
    'away_xG',
]

NUMERIC_COLUMNS = ['home_goals', 'away_goals', 'home_xG', 'away_xG']
DEDUP_COLUMNS = ['date', 'time', 'league', 'home_team', 'away_team']
DEFAULT_OUTPUT = Path('football_match_data.csv')
DEFAULT_BACKUP_DIR = Path('data_backups')
UNDERSTAT_URL = 'https://understat.com/league/{league}/{year}'
UNDERSTAT_API_URL = 'https://understat.com/getLeagueData/{league}/{year}'
USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/124.0 Safari/537.36'
)


@dataclass
class UpdateReport:
    previous_matches: int
    fetched_matches: int
    final_matches: int
    duplicate_matches_removed: int
    incomplete_matches_removed: int
    seasons_imported: list[str]
    latest_dates_by_league: dict[str, str]
    output: str
    backup: str | None
    dry_run: bool
    trained: bool
    training_metrics: dict | None


def default_end_year(today=None):
    today = today or datetime.now()
    return today.year if today.month >= 7 else today.year - 1


def decode_understat_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def understat_title(value):
    decoded = decode_understat_value(value)
    if isinstance(decoded, dict):
        return decoded.get('title') or decoded.get('name')
    return decoded


def understat_side_value(value, side):
    decoded = decode_understat_value(value)
    if isinstance(decoded, dict):
        return decoded.get(side)
    return decoded


def extract_understat_matches(html):
    matches = re.findall(r"JSON\.parse\('(.+?)'\)", html, flags=re.DOTALL)
    if not matches:
        raise ValueError('Could not find Understat JSON.parse payload in page HTML.')

    last_error = None
    for raw_payload in matches:
        try:
            decoded = raw_payload.encode('utf8').decode('unicode_escape')
            payload = json.loads(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            last_error = error
            continue
        matches = extract_matches_from_payload(payload)
        if matches:
            return matches

    raise ValueError(f'No match list found in Understat payloads: {last_error}')


def extract_matches_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ('dates', 'matches'):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []


def fetch_understat_matches(league, year, session=None, timeout=20, retries=2, pause=1.5, api_league=None):
    session = session or requests.Session()
    api_url = UNDERSTAT_API_URL.format(league=api_league or league, year=year)
    page_url = UNDERSTAT_URL.format(league=league, year=year)
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'X-Requested-With': 'XMLHttpRequest',
        'Referer': page_url,
    }
    errors = []

    for attempt in range(1, retries + 1):
        try:
            response = session.get(api_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            matches = extract_matches_from_payload(response.json())
            if not matches:
                raise ValueError('Understat API response did not include matches.')
            return matches
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            errors.append(f'API attempt {attempt}: {error}')

        try:
            response = session.get(page_url, headers={'User-Agent': USER_AGENT}, timeout=timeout)
            response.raise_for_status()
            return extract_understat_matches(response.text)
        except (requests.RequestException, ValueError) as error:
            errors.append(f'HTML fallback attempt {attempt}: {error}')
            if attempt < retries:
                time.sleep(pause * attempt)

    raise RuntimeError(f'Failed to fetch Understat data for {league} {year}: {" | ".join(errors)}')


def scrape_understat_range(start_year, end_year, timeout=20, retries=2, pause=1.5):
    if end_year < start_year:
        raise ValueError('end_year must be greater than or equal to start_year.')

    frames = []
    session = requests.Session()
    for year in range(start_year, end_year + 1):
        for understat_league, league_config in LEAGUES.items():
            print(f'Scraping Understat {understat_league} {year}...', flush=True)
            matches = fetch_understat_matches(
                understat_league,
                year,
                session=session,
                timeout=timeout,
                retries=retries,
                pause=pause,
                api_league=league_config['api'],
            )
            frame = pd.DataFrame(matches)
            frame['league'] = league_config['name']
            frame['season'] = f'{year}-{year + 1}'
            frames.append(frame)
            time.sleep(pause)

    if not frames:
        raise RuntimeError('No Understat frames were fetched.')
    return pd.concat(frames, ignore_index=True)


def transform_understat_frame(raw_df):
    df = raw_df.copy()
    if 'datetime' not in df.columns:
        raise ValueError('Understat data is missing the datetime column.')

    transformed = pd.DataFrame()
    parsed_datetime = pd.to_datetime(df['datetime'], errors='coerce')
    transformed['date'] = parsed_datetime.dt.strftime('%Y-%m-%d')
    transformed['time'] = parsed_datetime.dt.strftime('%H:%M:%S')
    transformed['league'] = df['league']
    transformed['season'] = df['season']
    transformed['home_team'] = df['h'].apply(understat_title)
    transformed['away_team'] = df['a'].apply(understat_title)
    transformed['home_goals'] = df['goals'].apply(lambda value: understat_side_value(value, 'h'))
    transformed['away_goals'] = df['goals'].apply(lambda value: understat_side_value(value, 'a'))
    transformed['home_xG'] = df['xG'].apply(lambda value: understat_side_value(value, 'h'))
    transformed['away_xG'] = df['xG'].apply(lambda value: understat_side_value(value, 'a'))

    transformed[NUMERIC_COLUMNS] = transformed[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')
    return transformed[REQUIRED_COLUMNS]


def validate_and_clean_data(df):
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {", ".join(missing_columns)}')

    clean_df = df[REQUIRED_COLUMNS].copy()
    before_incomplete = len(clean_df)
    clean_df['date'] = pd.to_datetime(clean_df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    parsed_time = pd.to_datetime(clean_df['time'].astype(str), format='%H:%M:%S', errors='coerce')
    missing_time = parsed_time.isna()
    if missing_time.any():
        parsed_time.loc[missing_time] = pd.to_datetime(
            clean_df.loc[missing_time, 'time'].astype(str),
            format='%H:%M',
            errors='coerce',
        )
    clean_df['time'] = parsed_time.dt.strftime('%H:%M:%S')
    clean_df[NUMERIC_COLUMNS] = clean_df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors='coerce')
    clean_df = clean_df.dropna(subset=REQUIRED_COLUMNS)
    incomplete_removed = before_incomplete - len(clean_df)

    before_dedup = len(clean_df)
    clean_df = clean_df.drop_duplicates(subset=DEDUP_COLUMNS, keep='last')
    duplicate_removed = before_dedup - len(clean_df)

    clean_df = clean_df.sort_values(['date', 'time', 'league', 'home_team', 'away_team']).reset_index(drop=True)
    if clean_df.empty:
        raise ValueError('No complete matches remained after cleaning.')

    return clean_df, {
        'incomplete_removed': int(incomplete_removed),
        'duplicate_removed': int(duplicate_removed),
    }


def count_existing_matches(output_path):
    if not output_path.exists():
        return 0
    return len(pd.read_csv(output_path))


def backup_existing_file(output_path, backup_dir=None):
    if not output_path.exists():
        return None
    backup_dir = backup_dir or output_path.parent / DEFAULT_BACKUP_DIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = backup_dir / f'{output_path.stem}_{timestamp}{output_path.suffix}'
    shutil.copy2(output_path, backup_path)
    return backup_path


def write_data(clean_df, output_path, dry_run=False):
    if dry_run:
        return None
    backup_path = backup_existing_file(output_path)
    clean_df.to_csv(output_path, index=False)
    return backup_path


def load_training_metrics(metrics_path=Path('model_evaluation.json')):
    if not metrics_path.exists():
        return None
    with metrics_path.open('r', encoding='utf-8') as file:
        return json.load(file)


def train_models():
    command = [sys.executable, 'train_honest_model.py']
    print(f'Running training command: {" ".join(command)}')
    subprocess.run(command, check=True)
    return load_training_metrics()


def summarize_data(clean_df):
    return {
        'seasons_imported': sorted(clean_df['season'].unique().tolist()),
        'latest_dates_by_league': clean_df.groupby('league')['date'].max().to_dict(),
    }


def build_report(
    previous_matches,
    fetched_matches,
    clean_df,
    clean_stats,
    output_path,
    backup_path=None,
    dry_run=False,
    trained=False,
    training_metrics=None,
):
    summary = summarize_data(clean_df)
    return UpdateReport(
        previous_matches=int(previous_matches),
        fetched_matches=int(fetched_matches),
        final_matches=int(len(clean_df)),
        duplicate_matches_removed=clean_stats['duplicate_removed'],
        incomplete_matches_removed=clean_stats['incomplete_removed'],
        seasons_imported=summary['seasons_imported'],
        latest_dates_by_league=summary['latest_dates_by_league'],
        output=str(output_path),
        backup=str(backup_path) if backup_path else None,
        dry_run=dry_run,
        trained=trained,
        training_metrics=training_metrics,
    )


def print_report(report):
    print('\nData update report')
    print('------------------')
    print(f'Output: {report.output}')
    print(f'Dry run: {report.dry_run}')
    print(f'Previous matches: {report.previous_matches}')
    print(f'Fetched matches: {report.fetched_matches}')
    print(f'Final matches: {report.final_matches}')
    print(f'Incomplete matches removed: {report.incomplete_matches_removed}')
    print(f'Duplicate matches removed: {report.duplicate_matches_removed}')
    print(f'Seasons imported: {", ".join(report.seasons_imported)}')
    print('Latest date by league:')
    for league, latest_date in sorted(report.latest_dates_by_league.items()):
        print(f'  - {league}: {latest_date}')
    print(f'Backup: {report.backup or "none"}')
    print(f'Training run: {report.trained}')
    if report.training_metrics:
        result_model = report.training_metrics.get('result_model', {})
        baselines = report.training_metrics.get('baselines', {})
        score_model = report.training_metrics.get('score_model', {})
        print('Training metrics:')
        print(f'  - Honest ML accuracy: {result_model.get("accuracy")}')
        print(f'  - Poisson baseline accuracy: {baselines.get("poisson_accuracy")}')
        print(f'  - Home goals MAE: {score_model.get("home_mae")}')
        print(f'  - Away goals MAE: {score_model.get("away_mae")}')


def update_data(
    start_year=2014,
    end_year=None,
    output_path=DEFAULT_OUTPUT,
    train=False,
    dry_run=False,
    timeout=20,
    retries=2,
    pause=1.5,
):
    output_path = Path(output_path)
    end_year = default_end_year() if end_year is None else end_year

    previous_matches = count_existing_matches(output_path)
    raw_df = scrape_understat_range(start_year, end_year, timeout=timeout, retries=retries, pause=pause)
    transformed_df = transform_understat_frame(raw_df)
    clean_df, clean_stats = validate_and_clean_data(transformed_df)
    backup_path = write_data(clean_df, output_path, dry_run=dry_run)

    training_metrics = None
    trained = False
    if train and not dry_run:
        training_metrics = train_models()
        trained = True

    report = build_report(
        previous_matches=previous_matches,
        fetched_matches=len(transformed_df),
        clean_df=clean_df,
        clean_stats=clean_stats,
        output_path=output_path,
        backup_path=backup_path,
        dry_run=dry_run,
        trained=trained,
        training_metrics=training_metrics,
    )
    print_report(report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Update football match data from Understat.')
    parser.add_argument('--start-year', type=int, default=2014)
    parser.add_argument('--end-year', type=int, default=None)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--train', dest='train', action='store_true', help='Retrain models after updating data.')
    parser.add_argument('--skip-train', dest='train', action='store_false', help='Do not retrain models after updating data.')
    parser.add_argument('--dry-run', action='store_true', help='Fetch and validate data without writing files or training.')
    parser.add_argument('--timeout', type=int, default=20)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--pause', type=float, default=1.5)
    parser.set_defaults(train=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        update_data(
            start_year=args.start_year,
            end_year=args.end_year,
            output_path=args.output,
            train=args.train,
            dry_run=args.dry_run,
            timeout=args.timeout,
            retries=args.retries,
            pause=args.pause,
        )
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
