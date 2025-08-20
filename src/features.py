import pandas as pd
import os

def add_rolling_features(input_path="data/clean_games.csv", output_path="data/features.csv"):
    """
    Adds rolling average features for each team based on the last N games.
    """

    os.makedirs("data", exist_ok=True)

    # Load cleaned data
    df = pd.read_csv(input_path, parse_dates=["GAME_DATE"])
    df = df.sort_values(["TEAM_FULL_NAME", "GAME_DATE"])

    # Stats to track
    stats = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3_PCT", "FT_PCT"]

    # Calculate rolling averages of the last 5 games
    for stat in stats:
        df[f"{stat}_rolling"] = (
            df.groupby("TEAM_FULL_NAME")[stat]
              .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )

    # Save features
    df.to_csv(output_path, index=False)
    print(f"Feature dataset saved with: {df.shape[0]} rows and {df.shape[1]} columns to {output_path}")

    return df


if __name__ == "__main__":
    add_rolling_features()
