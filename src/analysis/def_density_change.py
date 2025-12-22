import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from kloppy.domain import TrackingDataset

from ..metrics.def_density_change import ddc_merged_physical_data, metric_ddc
from ..data_loading.physical_data import get_physical_data_processed
from ..plot_functions.def_density_change import plot_scatter_ddc_distance

def def_density_change_analysis(
    dynamic_events_all: pd.DataFrame, 
    all_tracking: List[TrackingDataset], 
    all_metadata: List[Dict[str, Any]],
    data_path: str,
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0) -> pd.DataFrame:
    """
    Computes and merges defensive density change per 90 minutes with physical data for midfielders.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_tracking (list): List of tracking datasets for all matches.
        all_metadata (list): List of match metadata dictionaries.
        data_path (str): Path to the physical data file.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.    

    Returns:
        pd.DataFrame: Merged DataFrame with defensive density change and physical data for eligible players.
    """

    mid_pd = get_physical_data_processed(data_path)


    mid_obr_ddc, _ = metric_ddc(
        dynamic_events_all, 
        all_tracking, 
        all_metadata,
        min_matches,
        min_avg_minutes_played
    )

    mid_pd_merged = ddc_merged_physical_data(mid_obr_ddc, mid_pd)

    return mid_pd_merged

def a_ddc_distance_player(
    dynamic_events_all: pd.DataFrame, 
    all_tracking: List[TrackingDataset], 
    all_metadata: List[Dict[str, Any]],
    data_path: str,
    season: Optional[str] = "2024/2025",
    competition: Optional[str] = "Australian A-League",
    total_matches: Optional[int] = 10,
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0
    ) -> None:
    """
    Does the analysis of defensive density change per 90 minutes vs distance tip per 90 minutes.
    Generates a scatter plot of the results.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_tracking (list): List of tracking datasets for all matches.
        all_metadata (list): List of match metadata dictionaries.
        data_path (str): Path to the physical data file.
        season (str, optional): Season label for the plot.
        competition (str, optional): Competition label for the plot.
        total_matches (int, optional): Total number of matches considered for the plot.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.    
    """

    df_ddc_pd = def_density_change_analysis(
        dynamic_events_all, 
        all_tracking, 
        all_metadata,
        data_path,
        min_matches,
        min_avg_minutes_played
    )

    plot_scatter_ddc_distance(
        df_ddc_pd,
        season,
        competition,
        total_matches,
        min_matches,
        min_avg_minutes_played
    )