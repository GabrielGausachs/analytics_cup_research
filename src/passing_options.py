from .preprocessing import filter_elegible_players
from .tracking_functions import find_frame_start_end, get_player_coordinates
from typing import List, Dict, Any, Optional
import pandas as pd
from kloppy.domain import TrackingDataset


def po_created(mid_obr: pd.DataFrame, all_tracking: List[TrackingDataset]) -> pd.DataFrame:
    """
    Calculate passing options created during midfield off-ball runs.

    Args:
        mid_obr (pd.DataFrame): DataFrame of midfield off-ball events.
        all_tracking (list): List of tracking datasets for all matches.
    
    Returns:
        pd.DataFrame: The input DataFrame with an additional column 'passing_options_created'
            representing the number of passing options created during the off-ball event.
    """
    mid_obr = mid_obr.copy()
    
    for row in mid_obr.itertuples():
        
    pass

def metric_pocreated(
    dynamic_events_all: pd.DataFrame, 
    all_tracking: List[TrackingDataset], 
    all_metadata: List[Dict[str, Any]],
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0)-> pd.DataFrame:
    """
    Computes passing options created influenced by the run for midfielders.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_tracking (list): List of tracking datasets for all matches.
        all_metadata (list): List of match metadata dictionaries.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.
    
    Returns:
        pd.DataFrame: DataFrame with passing options created per 90 minutes for eligible players.
    """

    # Filter eligible players
    mid_obr_filtered, eligible_players = filter_elegible_players(
        dynamic_events_all, 
        all_metadata,
        min_matches,
        min_avg_minutes_played
    )

    # Calculate passing options created
    mid_obr_filtered = po_created(mid_obr_filtered, all_tracking)