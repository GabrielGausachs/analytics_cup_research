import pandas as pd
from typing import List, Dict, Any, Optional
from kloppy.domain import TrackingDataset

from ..plot_functions.space_created import animate_run_by_event_id
from ..metrics.space_created import metric_sc

def a_space_created(
    dynamic_events_all: pd.DataFrame, 
    all_metadata: List[Dict[str, Any]], 
    all_tracking: List[TrackingDataset], 
    event_id: str,
    min_matches: Optional[int] = 2,
    min_avg_minutes_played: Optional[int] = 40,
    csv_path: Optional[str] = None) -> None:

    """
    Animate space created for a specific off-ball run event by event_id.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_metadata (list): List of match metadata dictionaries.
        all_tracking (list): List of tracking datasets for all matches.
        event_id (str): The event ID of the off-ball run to animate.
        min_matches (int, optional): Minimum number of matches a player must have played to be included. Defaults to 2.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included. Defaults to 40.
        csv_path (str, optional): Path to a CSV file containing precomputed space created data. If provided, this will be used instead of computing from scratch. Defaults to None.
    """

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        _, df = metric_sc(dynamic_events_all, all_tracking, all_metadata, min_matches=min_matches, min_avg_minutes_played=min_avg_minutes_played)

    return animate_run_by_event_id(event_id, df, all_tracking)