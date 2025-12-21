from datetime import timedelta
from kloppy.domain import TrackingDataset
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .helpers import whisker_bounds_numpy


def match_minutes_played(match_tracking: TrackingDataset) -> float:
    """
    Computes the total number of minutes played in a match based on the
    tracking dataset's period timestamps.

    The function iterates over all periods in the match metadata and sums
    their durations. It then returns the total played time in minutes.

    Args:
        match_tracking (TrackingDataset): A kloppy tracking dataset.
            It must contain `metadata.periods`, where each period has
            `start_timestamp` and `end_timestamp` attributes.

    Returns:
        float: Total match duration in minutes.
    """

    total_duration = timedelta(0)

    for period in match_tracking.metadata.periods:
        period_duration = period.end_timestamp - period.start_timestamp
        total_duration += period_duration

    total_minutes = total_duration.total_seconds() / 60

    return total_minutes


def player_minutes_per_match(all_metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Return minutes played per player per match.

    Args:
        all_metadata (list): List of match metadata dictionaries.

    Returns:
        pd.DataFrame: DataFrame with columns
            ['match_id', 'player_id', 'minutes_played']
    """

    records = []

    for metadata in all_metadata:
        match_id = metadata.get("id")

        for player in metadata.get("players", []):
            player_id = player.get("id")
            team_id = player.get("team_id")

            if player_id is None:
                continue

            total_time = player.get("playing_time", {}).get("total", {})

            if total_time is None:
                continue
            else:
                minutes = total_time.get("minutes_played", 0)

            records.append({
                "match_id": match_id,
                "player_id": player_id,
                "team_id": team_id,
                "minutes_played": minutes
            })

    return pd.DataFrame(records)


def midfielders_obr(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Extract off-ball runs performed by midfielders,
    keeping only events with matched possession start and end.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events data for all matches.

    Returns:
        pd.DataFrame: Filtered DataFrame containing midfielder off-ball events.
    """
    # Get off-ball events
    off_ball_events = dynamic_events_all[dynamic_events_all["event_type_id"] == 1]

    # Get only off ball events from midfielders
    positions_mid = [9,10,11,12,13,14,15]
    mid_obr = off_ball_events[off_ball_events["player_position_id"].isin(positions_mid)].copy()

    # For every obe, column id equals event_id_match_id
    mid_obr["id"] = mid_obr["event_id"].astype(str) + "_" + mid_obr["match_id"].astype(str)
    mid_obr = mid_obr.reset_index(drop=True)
    # Data matching
    mid_obr = mid_obr[
            (mid_obr["is_player_possession_start_matched"] == True) &
            (mid_obr["is_player_possession_end_matched"] == True)
        ]
    
    return mid_obr

def preprocess_physical_data(physical_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess physical data to compute distance covered per 90 minutes for midfield players.

    Args:
        physical_data (pd.DataFrame): DataFrame containing physical data for players.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with distance covered per 90 minutes for midfield players.
    """

    # Filter only midfielders
    physical_mid = physical_data[physical_data["position_group"] == "Midfield"]

    # Select columns of interest
    cols = [
        "player_id", "player_short_name", "team_name", "season_name", "total_metersperminute_full_tip",
    ]
    physical_mid = physical_mid[cols]


    # Compute per 90-minute metrics
    physical_mid["distance_tip_per90"] = physical_mid["total_metersperminute_full_tip"] * 90

    # Sort by distance per 90
    physical_mid = physical_mid.sort_values(by="distance_tip_per90", ascending=False).reset_index(drop=True)

    return physical_mid

def filter_eligible_players(
    dynamic_events_all: pd.DataFrame, 
    all_metadata: List[Dict[str, Any]],
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter by midfielders who have played at least min_matches and min_avg_minutes_played.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_metadata (list): List of match metadata dictionaries.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.
    
    Returns:
        tuple: (mid_obr_filtered, eligible_players)
            - mid_obr_filtered (pd.DataFrame): Filtered DataFrame containing midfielder off-ball events.
            - eligible_players (pd.DataFrame): DataFrame of eligible players with their match and minutes played stats.
    """
    
    mid_obr = midfielders_obr(dynamic_events_all)

    player_minutes_df = player_minutes_per_match(all_metadata)

    # Get eligible players based on min_matches and min_avg_minutes_played
    eligible_players = (
        player_minutes_df.groupby("player_id")
        .agg(
            matches=("match_id", "nunique"),
            avg_minutes=("minutes_played", "mean"),
            total_minutes=("minutes_played", "sum"),
        )
        .reset_index()
    )

    eligible_players = eligible_players[
        (eligible_players["matches"] >= min_matches) &
        (eligible_players["avg_minutes"] >= min_avg_minutes_played)
    ]

    # Filter mid_obr to include only eligible players
    mid_obr_filtered = mid_obr[mid_obr["player_id"].isin(eligible_players["player_id"])]

    return mid_obr_filtered, eligible_players


def remove_outliers(df: pd.DataFrame, cols: List[str], subtype_col: str = "event_subtype") -> pd.DataFrame:
    """
    Removes row-wise outliers per event subtype using IQR (boxplot) method.

    Args:
        df (pd.DataFrame): DataFrame with event-level metrics.
        cols (List[str]): List of numeric columns to check for outliers.
        subtype_col (str): Column indicating the event subtype.

    Returns:
        pd.DataFrame: DataFrame with outlier rows removed.
    """

    X = df[cols].to_numpy()
    subtypes = df[subtype_col].to_numpy()
    event_subtype_arrays = {s: X[subtypes == s] for s in np.unique(subtypes)}
    

    col_idx = {c: i for i, c in enumerate(cols)}

    outlier_rows_per_subtype = {}
    for subtype, arr in event_subtype_arrays.items():
        row_outlier_mask = np.zeros(arr.shape[0], dtype=bool)
        for idx in col_idx.values():
            lower, upper = whisker_bounds_numpy(arr[:, idx])
            row_outlier_mask |= (arr[:, idx] < lower) | (arr[:, idx] > upper)
        row_indices = np.where(subtypes == subtype)[0]
        outlier_rows_per_subtype[subtype] = row_indices[row_outlier_mask]

    # Combine all outlier indices
    if len(outlier_rows_per_subtype) > 0:
        all_outlier_indices = np.concatenate(list(outlier_rows_per_subtype.values()))
    else:
        all_outlier_indices = np.array([], dtype=int)

    # Remove outliers
    df_clean = df.drop(index=df.index[all_outlier_indices]).copy()
    return df_clean
