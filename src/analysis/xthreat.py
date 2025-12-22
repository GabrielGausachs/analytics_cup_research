import pandas as pd
from typing import List, Dict, Any, Optional

from ..metrics.xthreat import obr_xthreat
from ..plot_functions.xthreat import plot_violin_xthreat
from ..utils.helpers import z_score


def a_xthreat_per_run_group(
    dynamic_events_all: pd.DataFrame,
    all_metadata: List[Dict[str, Any]],
    season: Optional[str] = "2024/2025",
    competition: Optional[str] = "Australian A-League",
    total_matches: Optional[int] = 10,
    min_matches: Optional[int] = 0,
    min_avg_minutes_played: Optional[int] = 0
    ) -> None:
    """
    Computes xThreat contribution per run group for midfielders off-ball runs,
    calculates z-scores and plot the results.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
        all_metadata (List[Dict[str, Any]]): List of metadata dictionaries for all matches.
        season (Optional[str]): Season string for annotation.
        competition (Optional[str]): Competition string for annotation.
        total_matches (Optional[int]): Number of matches for annotation.
        min_matches (Optional[int]): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (Optional[int]): Minimum average minutes played per match for a player to be included.
    """

    _ , df_rungroup = obr_xthreat(
        dynamic_events_all,
        all_metadata,
        min_matches,
        min_avg_minutes_played)
    
    df_rungroup = z_score(
        df_rungroup,
        value_col="xthreat_per90",
        group_col="run_group",)
    
    plot_violin_xthreat(
        df_rungroup,
        season=season,
        competition=competition,
        total_matches=total_matches,
        min_matches=min_matches,
        min_avg_minutes_played=min_avg_minutes_played)