import pandas as pd
import os
import time
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams

def collect_all_games(output_path="data/all_games.csv"):
    """
    Collects all available games for all NBA teams and saves to CSV.
    """

    os.makedirs("data", exist_ok=True)

    all_games = pd.DataFrame()
    nba_teams = teams.get_teams()
    print(f"Found {len(nba_teams)} teams.")

    for i, team in enumerate(nba_teams, start=1):
        print(f"[{i}/{len(nba_teams)}] Getting games for {team['full_name']}")

        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team["id"])
            games = gamefinder.get_data_frames()[0]
            games["TEAM_FULL_NAME"] = team["full_name"]

            all_games = pd.concat([all_games, games], ignore_index=True)

            time.sleep(0.6)

        except Exception as e:
            print(f"Could not get games for {team['full_name']}: {e}")

    if not all_games.empty:
        all_games.to_csv(output_path, index=False)
        print(f"Saved {all_games.shape[0]} rows and {all_games.shape[1]} columns to {output_path}")
    else:
        print("No data has been collected")
