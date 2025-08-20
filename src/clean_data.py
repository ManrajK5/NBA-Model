import pandas as pd
import os

def clean_games_data(input_path="data/all_games.csv", output_path="data/clean_games.csv"):
    """
    Cleans the raw NBA games data and saves a simplified version
    ready for feature engineering.
    """

    # Ensure data folder exists
    os.makedirs("data", exist_ok=True)

    # Load raw data
    df = pd.read_csv(input_path)
    print(f"Loaded raw data with {df.shape[0]} rows and {df.shape[1]} columns.")

    # Drop rows with missing vals
    critical_cols = ["MATCHUP", "WL", "PTS", "REB", "AST"]
    df = df.dropna(subset=critical_cols)
    print(f"After dropping incomplete rows: {df.shape[0]} rows remain.")

    # Keep useful columns
    keep_cols = [
        "TEAM_FULL_NAME", "TEAM_ABBREVIATION", "GAME_DATE", "MATCHUP", "WL",
        "PTS", "REB", "AST", "STL", "BLK", "TOV", "PF",
        "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"
    ]
    df = df[keep_cols]

    # Convert WL column (W/L) into numeric (1/0)
    df["WIN"] = df["WL"].apply(lambda x: 1 if x == "W" else 0)

    # Create home/away flag
    df["HOME_GAME"] = df["MATCHUP"].apply(lambda x: 1 if isinstance(x, str) and "vs." in x else 0)

    # Drop old WL column
    df = df.drop(columns=["WL"])

    # Convert GAME_DATE to datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset with: {df.shape[0]} rows and {df.shape[1]} columns to {output_path}")

    return df


if __name__ == "__main__":
    clean_games_data()
