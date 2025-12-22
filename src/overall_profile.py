from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from .utils.preprocessing import filter_eligible_players
from .offball_wrappers import obr_xthreat
from .density_change import metric_ddc
from kloppy.domain import TrackingDataset
from .space_creation import metric_sc
from .data_loading.physical_data import get_physical_data_processed
from .plots import plot_multiple_radar_plots_players_overall

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
    df_sc,_ = metric_sc(
        dynamic_events_all,
        all_tracking,
        all_metadata,
        min_matches=min_matches,
        min_avg_minutes_played=min_avg_minutes_played)
    
    # Distance covered metrics for midfielders performing off-ball runs
    df_physical = get_physical_data_processed(data_path)


    # Prepare the data for the radar plot
    df_xthreat = df_xthreat.reset_index()
    df_ddc = df_ddc.reset_index()
    df_sc = df_sc.reset_index()
    df_physical = df_physical.reset_index()

    ddc_bu = (
        df_ddc
        .query("rungroup == 'Build Up'")
        [["player_id", "def_density_change_per90min"]]
        .rename(columns={"def_density_change_per90min": "ddc_build_up"})
    )

    ddc_prog = (
        df_ddc
        .query("rungroup == 'Progression'")
        [["player_id", "def_density_change_per90min"]]
        .rename(columns={"def_density_change_per90min": "ddc_progression"})
    )

    xt_prog = (
        df_xthreat
        .query("run_group == 'Progression'")
        [["player_id", "xthreat_per90"]]
        .rename(columns={"xthreat_per90": "xT_progression"})
    )

    xt_direct = (
        df_xthreat
        .query("run_group == 'Direct'")
        [["player_id", "xthreat_per90"]]
        .rename(columns={"xthreat_per90": "xT_direct"})
    )

    # Get all player_ids present in the other dataframes
    players_to_keep = set(ddc_bu['player_id']) | set(ddc_prog['player_id']) | set(xt_prog['player_id']) | set(xt_direct['player_id']) | set(df_sc['player_id'])

    # Filter df_physical
    dist_poss_filtered = df_physical[df_physical['player_id'].isin(players_to_keep)][['player_id', 'distance_tip_per90']].rename(columns={'distance_tip_per90': 'dist_poss_90'})
    
    radar_df = (
        ddc_bu
        .merge(df_sc, on="player_id", how="outer", validate="one_to_one")
        .merge(ddc_prog, on="player_id", how="outer", validate="one_to_one")
        .merge(xt_prog, on="player_id", how="outer", validate="one_to_one")
        .merge(xt_direct, on="player_id", how="outer", validate="one_to_one")
        .merge(dist_poss_filtered, on="player_id", how="outer", validate="one_to_one")
    )

    # also add player names from df_xthreat
    player_names = df_xthreat[['player_id', 'player_name']].drop_duplicates()
    radar_df = radar_df.merge(player_names, on='player_id', how='left', validate='one_to_one')

    return radar_df


def a_obr_radar_all(
    dynamic_events_all: pd.DataFrame,
    all_metadata: List[Dict[str, Any]],
    all_tracking: List[TrackingDataset],
    data_path: str, 
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0,
    player_names: Optional[List[str]] = None,
    csv_path: Optional[str] = None
    ) -> pd.DataFrame:
    """
    Compute and aggregate various metrics for A-league midfielders performing off-ball runs.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_metadata (list): List of match metadata dictionaries.
        all_tracking (list): List of tracking datasets for all matches.
        data_path (str): The path to the data directory.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.
        player_names (list, optional): List of player names to highlight in the radar plots.
        csv_path (str, optional): Path to a CSV file containing precomputed radar data. If provided, this will be used instead of computing from scratch. Defaults to None.
    """

    if csv_path:
        radar_df = pd.read_csv(csv_path)
    else:
        radar_df = obr_radar_all(
            all_metadata,
            all_tracking,
            dynamic_events_all,
            data_path,
            min_matches,
            min_avg_minutes_played
        )

    plot_multiple_radar_plots_players_overall(radar_df, players_names=player_names)
