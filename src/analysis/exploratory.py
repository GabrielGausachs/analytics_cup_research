from typing import List, Optional
import pandas as pd
from kloppy.domain import TrackingDataset

from src.offball_wrappers import pct_runs_subtype_phase, obr_per_subtype_per90min, obr_per_subtype_per_phase
from ..plot_functions.exploratory import plot_total_and_untargeted_per90, subtype_phase_bubble_plot


def a_obr_per_subtype(
    all_tracking: List[TrackingDataset],
    dynamic_events_all: pd.DataFrame,
    season: Optional[str] = "2024/2025",
    competition: Optional[str] = "Australian A-League",
    total_matches: Optional[int] = 10
    ) -> None:
    """
    Compute off-ball runs statistics per subtype normalized per 90 minutes
    and plot total and untargeted counts.

    Args:
        all_tracking (List[TrackingDataset]): List of tracking datasets for all matches.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
        season (Optional[str]): Season string for labeling the plot (e.g., "2024/2025").
        competition (Optional[str]): Competition name for labeling the plot.
        total_matches (Optional[int]): Number of matches, for plot annotation.
    """
    df = obr_per_subtype_per90min(all_tracking, dynamic_events_all)
    plot_total_and_untargeted_per90(
        df,
        season=season,
        competition=competition,
        total_matches=total_matches
    )


def a_obr_per_subtype_per_phase(
    dynamic_events_all: pd.DataFrame,
    season: Optional[str] = "2024/2025",
    competition: Optional[str] = "Australian A-League",
    total_matches: Optional[int] = 10
    ) -> None:
    """
    Compute off-ball runs statistics per subtype per phase, convert to percentages,
    and plot as a bubble chart.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
        season (Optional[str]): Season string for annotation.
        competition (Optional[str]): Competition string for annotation.
        total_matches (Optional[int]): Total matches for annotation.
    """
    df_phase = obr_per_subtype_per_phase(dynamic_events_all)
    df_pct = pct_runs_subtype_phase(df_phase)
    subtype_phase_bubble_plot(
        df_pct,
        season=season,
        competition=competition,
        total_matches=total_matches
    )