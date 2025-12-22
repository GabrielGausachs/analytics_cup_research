import pandas as pd
from typing import List, Dict, Any, Optional
from kloppy.domain import TrackingDataset

from ..plot_functions.radar_plots import plot_multiple_radar_plots_teams, plot_multiple_radar_plots_players
from ..metrics.obr_wrappers import obr_per_subtype_per_team_per90min, obr_per_subtype_per_player_per90min

def a_obr_per_subtype_per_team(
    all_tracking: List[TrackingDataset],
    dynamic_events_all: pd.DataFrame,
    teams_shortnames: Optional[List[str]] = None,
    season: Optional[str] = "2024/2025",
    competition: Optional[str] = "Australian A-League",
    total_matches: Optional[int] = 10
    ) -> None:
    """
    Computes off-ball runs stats per subtype and team per 90 minutes of play and plots a radar plot for selected teams.

    Args:
        all_tracking (List[TrackingDataset]): List of tracking datasets for all matches.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
        teams_shortnames (Optional[List[str]]): List of team shortnames to plot.
        season (Optional[str]): Season string for annotation.
        competition (Optional[str]): Competition string for annotation.
        total_matches (Optional[int]): Number of matches for annotation.
    """

    df_pivot, df_percentile = obr_per_subtype_per_team_per90min(all_tracking, dynamic_events_all)

    if teams_shortnames is not None:
        plot_multiple_radar_plots_teams(
            df_pivot=df_pivot,
            df_percentile=df_percentile,
            teams_shortnames=teams_shortnames,
            season=season,
            competition=competition,
            total_matches=total_matches)
    else:
        raise ValueError("Please provide a list of team shortnames to plot radar plots.")
    

def a_obr_per_subtype_per_player(
    all_metadata: List[Dict[str, Any]],
    dynamic_events_all: pd.DataFrame,
    players_names: Optional[List[str]] = None,
    season: Optional[str] = "2024/2025",
    competition: Optional[str] = "Australian A-League",
    total_matches: Optional[int] = 10,
    min_matches: Optional[int] = 0,
    min_avg_minutes_played: Optional[int] = 0
    ) -> None:
    """
    Computes off-ball runs stats per subtype and player per 90 minutes of play and plots a radar plot for selected players.

    Args:
        all_metadata (List[Dict[str, Any]]): List of metadata dictionaries for all matches.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
        players_names (Optional[List[str]]): List of player names to plot.
        season (Optional[str]): Season string for annotation.
        competition (Optional[str]): Competition string for annotation.
        total_matches (Optional[int]): Number of matches for annotation.
        min_matches (Optional[int]): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (Optional[int]): Minimum average minutes played per match for a player to be included.
    """

    df_pivot, df_percentile = obr_per_subtype_per_player_per90min(all_metadata, dynamic_events_all, min_matches, min_avg_minutes_played)

    if players_names is not None:
        plot_multiple_radar_plots_players(
            df_pivot=df_pivot,
            df_percentile=df_percentile,
            players_names=players_names,
            season=season,
            competition=competition,
            total_matches=total_matches,
            min_matches=min_matches,
            min_avg_minutes_played=min_avg_minutes_played)
    else:
        raise ValueError("Please provide a list of player names to plot radar plots.")