from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from .preprocessing import filter_eligible_players
from .offball_wrappers import obr_xthreat
from .density_change import metric_ddc
from kloppy.domain import TrackingDataset
from .space_creation import metric_sc
from .physical_data import get_physical_data_processed

def obr_radar_all(
    all_metadata: List[Dict[str, Any]],
    all_tracking: List[TrackingDataset],
    dynamic_events_all: pd.DataFrame,
    data_path: str, 
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0
    ) -> pd.DataFrame:
    """
    Compute and aggregate various metrics for midfielders performing off-ball runs.

    Args:
        all_metadata (list): List of match metadata dictionaries.
        all_tracking (list): List of tracking datasets for all matches.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        data_path (str): The path to the data directory.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.

    Returns:
        pd.DataFrame: DataFrame containing aggregated metrics for eligible midfielders.
    """


    # Get xthreat metrics for midfielders performing off-ball runs
    _,df_xthreat = obr_xthreat(
        dynamic_events_all, 
        all_metadata, 
        min_matches=min_matches, 
        min_avg_minutes_played=min_avg_minutes_played)

    # Get defensive density change metrics for midfielders performing off-ball runs
    _,df_ddc = metric_ddc(
        dynamic_events_all, 
        all_tracking, 
        all_metadata, 
        min_matches=min_matches, 
        min_avg_minutes_played=min_avg_minutes_played)

    # Get space created metrics for midfielders performing off-ball runs
    _,df_sc = metric_sc(
        dynamic_events_all,
        all_tracking,
        all_metadata,
        min_matches=min_matches,
        min_avg_minutes_played=min_avg_minutes_played)
    
    # Distance covered metrics for midfielders performing off-ball runs
    df_physical = get_physical_data_processed(data_path)

    # Merge all metrics into a single DataFrame by player_id
    mid_obr_grouped = df_xthreat.merge(
        df_ddc,
        on="player_id",
        how="outer"
    ).merge(
        df_sc,
        on="player_id",
        how="outer"
    ).merge(
        df_physical,
        on="player_id",
        how="outer"
    )

    return mid_obr_grouped



