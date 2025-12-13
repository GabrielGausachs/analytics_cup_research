from .preprocessing import midfielders_obr, player_minutes_per_match
from .tracking_functions import find_frame_start_end, get_player_coordinates
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from kloppy.domain import TrackingDataset
from scipy.spatial.distance import cdist


def def_density_change(mid_obr: pd.DataFrame, all_tracking: List[TrackingDataset]) -> pd.DataFrame:
    """
    Calculates the change in average defensive density around midfield off-ball events.

    Args:
        mid_obr (pd.DataFrame): DataFrame of midfield off-ball events.
        all_tracking (list): List of tracking datasets for all matches.
    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'def_density_change'
            representing the change in average defensive density from start to end of the event.
    """

    mid_obr = mid_obr.copy()
    mid_obr["def_density_change"] = np.nan

    for row in mid_obr.itertuples():
        # Find start and end frames
        start_frame, end_frame = find_frame_start_end(row, all_tracking)
        if start_frame is None or end_frame is None:
            # remove row from mid_obr to avoid unreliable analysis
            mid_obr.drop(index=row.Index, inplace=True)
            continue
        
        # Get player coordinates at start frame
        player_coord = get_player_coordinates(start_frame, str(row.player_id))

        if player_coord is None:
            print(f"Player {row.player_id} not found in start frame of match {row.match_id}")

        # Start frame distances
        opponents_start = np.array([[c.x, c.y] for p, c in start_frame.players_coordinates.items()
                                    if p.team.team_id != row.team_id])
        
        if len(opponents_start) > 0:
            distances_start = cdist([player_coord], opponents_start).flatten()
            close_start = distances_start[distances_start <= 10]
            avg_distance_start = close_start.mean() if len(close_start) > 0 else 10.0
        else:
            print(f"No opponents found in start frame of match {row.match_id}")
            avg_distance_start = 10.0

        # Get player coordinates at end frame
        player_coord = get_player_coordinates(end_frame, str(row.player_id))

        if player_coord is None:
            print(f"Player {row.player_id} not found in end frame of match {row.match_id}")
            continue

        # End frame distances
        opponents_end = np.array([[c.x, c.y] for p, c in end_frame.players_coordinates.items()
                                if p.team.team_id != row.team_id])
        
        if len(opponents_end) > 0:
            distances_end = cdist([player_coord], opponents_end).flatten()
            close_end = distances_end[distances_end <= 10]
            avg_distance_end = close_end.mean() if len(close_end) > 0 else 10.0
        else:
            print(f"No opponents found in end frame of match {row.match_id}")
            avg_distance_end = 10.0

        # Store results in mid_obr DataFrame
        mid_obr.at[row.Index, 'def_density_change'] = avg_distance_end - avg_distance_start
        
    return mid_obr

def metric_ddc(
    dynamic_events_all: pd.DataFrame, 
    all_tracking: List[TrackingDataset], 
    all_metadata: List[Dict[str, Any]],
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0)-> pd.DataFrame:
    """
    Computes defensive density change per 90 minutes for midfielders performing off-ball runs.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_tracking (list): List of tracking datasets for all matches.
        all_metadata (list): List of match metadata dictionaries.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.    

    Returns:
        pd.DataFrame: DataFrame with defensive density change per 90 minutes for eligible players.
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

    # Calculate defensive density change
    mid_obr_filtered = def_density_change(mid_obr_filtered, all_tracking)

    # Merge to get total minutes played per player
    mid_obr_merged = mid_obr_filtered.merge(
        eligible_players[["player_id", "total_minutes"]],
        on="player_id",
        how="left"
    )

    return mid_obr_merged
