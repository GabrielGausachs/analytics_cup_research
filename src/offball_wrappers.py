import pandas as pd
from kloppy.domain import TrackingDataset

from .plots import plot_total_and_untargeted_per90, subtype_phase_bubble_plot, plot_multiple_radar_plots
from .preprocessing import match_minutes_played
from typing import List, Optional
from .aggregates import off_ball_event_agg, normalize_per90min

phase_order = [
        "build_up",
        "direct",
        "quick_break",
        "transition",
        "create",
        "finish",
        "chaotic",
        "disruption",
        "set_play"
]

subtypes_order = [
        "cross_receiver",
        "behind",
        "run_ahead_of_the_ball",
        "overlap",
        "underlap",
        "support",
        "coming_short",
        "dropping_off",
        "pulling_half_space",
        "pulling_wide"
]


def obr_per_subtype(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball runs statistics per event subtype.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns:['event_subtype', 'total', 'targeted', 'untargeted']
    """
    return off_ball_event_agg(dynamic_events_all, group_by=["event_subtype"])


def obr_per_subtype_per90min(all_tracking: List[TrackingDataset], dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball runs stats per subtype normalized per 90 minutes of play.

    Args:
        all_tracking (List[TrackingDataset]): List of tracking datasets for all matches.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
    
    Returns:
        pd.DataFrame: DataFrame with off-ball runs per subtype normalized per 90 minutes.
    """

    total_minutes = sum(match_minutes_played(match) for match in all_tracking)
    agg_df = obr_per_subtype(dynamic_events_all)
    return normalize_per90min(
        agg_df,
        total_minutes=total_minutes,
        columns_to_normalize=["total", "targeted", "untargeted"]
    )


def obr_per_subtype_per_phase(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the total number of off-ball runs per subtype per phase of play.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns:['team_in_possession_phase_type', 'event_subtype', 'total']
    """
    return off_ball_event_agg(
        dynamic_events_all,
        group_by=["team_in_possession_phase_type", "event_subtype"]
    )[["team_in_possession_phase_type", "event_subtype", "total"]]


def pct_runs_subtype_phase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares off-ball runs data for bubble plot visualization.

    Args:
        df (pd.DataFrame): DataFrame with columns ['team_in_possession_phase_type', 'event_subtype', 'total']

    Returns:
        pd.DataFrame: Pivoted DataFrame with percentages per subtype per phase.
    """

    # Pivot table
    df_pivot = df.pivot_table(
        index='event_subtype', columns='team_in_possession_phase_type', values='total', fill_value=0
    )

    # Ensure consistent order
    df_pivot = df_pivot.reindex(index=subtypes_order, columns=phase_order)

    # Convert to percentages per phase
    df_percent = df_pivot.divide(df_pivot.sum(axis=0), axis=1) * 100

    return df_percent

def obr_per_subtype_per_team(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball runs statistics per event subtype and team.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns:['event_subtype', 'team_id', 'team_shortname', 'total', 'targeted', 'untargeted']
    """

    return off_ball_event_agg(dynamic_events_all, group_by=["event_subtype", "team_id","team_shortname"])


def obr_per_subtype_per_team_per90min(all_tracking: List[TrackingDataset], dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball runs stats per subtype and team per 90 minutes of play.

    Args:
        all_tracking (List[TrackingDataset]): List of tracking datasets for all matches.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
    
    Returns:
        pd.DataFrame: Pivoted DataFrame with off-ball runs per subtype per team normalized per 90 minutes.
        pd.DataFrame: Percentile DataFrame with off-ball runs per subtype per team normalized per 90 minutes.

    """

    # Calculate total minutes per team across all matches
    team_minutes = {}
    for match in all_tracking:
        match_duration = match_minutes_played(match)
        for team in match.metadata.teams:
            if team.team_id not in team_minutes:
                team_minutes[team.team_id] = 0
            team_minutes[team.team_id] += match_duration

    # Get the off ball runs per subtype per team
    agg_df = obr_per_subtype_per_team(dynamic_events_all)

    # Add total minutes per team to the DataFrame
    agg_df["total_minutes"] = agg_df["team_id"].map(team_minutes)

    # Normalize per 90 minutes
    agg_df["runs_per90min"] = agg_df.apply(
        lambda row: (row["total"] / row["total_minutes"] * 90) if row["total_minutes"] > 0 else 0, axis=1
    )

    # Pivot the DataFrame to have subtypes as columns and teams as rows
    df_pivot = agg_df.pivot(
        index='team_shortname', 
        columns='event_subtype', 
        values='runs_per90min'
        ).fillna(0)

    # Calculate percentiles
    percentile_df = df_pivot.rank(pct=True) * 100

    return df_pivot, percentile_df




# ------------------------------
# High-level convenience plotting functions
# ------------------------------

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

def a_obr_per_subtype_per_team(
    all_tracking: List[TrackingDataset],
    dynamic_events_all: pd.DataFrame,
    teams_shortnames: Optional[List[str]] = None,
    season: Optional[str] = "2024/2025",
    competition: Optional[str] = "Australian A-League",
    total_matches: Optional[int] = 10
    ) -> pd.DataFrame:
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
        plot_multiple_radar_plots(
            df_pivot=df_pivot,
            df_percentile=df_percentile,
            teams_shortnames=teams_shortnames,
            season=season,
            competition=competition,
            total_matches=total_matches)
    else:
        raise ValueError("Please provide a list of team shortnames to plot radar plots.")
