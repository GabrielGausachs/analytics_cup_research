import pandas as pd
from kloppy.domain import TrackingDataset
from typing import List, Optional, Tuple, Dict, Any

from ..utils.helpers import entropy
from ..utils.preprocessing import filter_eligible_players, match_minutes_played
from ..utils.aggregates import off_ball_event_agg, normalize_per90min


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


def obr_per_subtype_per_team_per90min(all_tracking: List[TrackingDataset], dynamic_events_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

def obr_per_subtype_per_player(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball runs statistics per event subtype and player.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns:['event_subtype', 'player_id', 'player_name', 'total', 'targeted', 'untargeted']
    """

    return off_ball_event_agg(dynamic_events_all, group_by=["event_subtype", "player_id","player_name"])

def obr_per_subtype_per_player_per90min(
    all_metadata: List[Dict[str, Any]], 
    dynamic_events_all: pd.DataFrame, 
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes off-ball runs stats per subtype and player per 90 minutes of play.

    Args:
        all_metadata (List[Dict[str, Any]]): List of match metadata dictionaries.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
    
    Returns:
        pd.DataFrame: Pivoted DataFrame with off-ball runs per subtype per player normalized per 90 minutes.
        pd.DataFrame: Percentile DataFrame with off-ball runs per subtype per player normalized per 90 minutes.

    """

    # Get eligible players based on min_matches and min_avg_minutes_played
    _, eligible_players = filter_eligible_players(
        dynamic_events_all,
        all_metadata,
        min_matches,
        min_avg_minutes_played
    )

    # Get the off ball runs per subtype per player
    agg_df = obr_per_subtype_per_player(dynamic_events_all)

    # Add total minutes per player to the DataFrame
    agg_df = agg_df.merge(
        eligible_players[["player_id", "total_minutes"]],
        on="player_id",
        how="inner"
    )

    # Normalize per 90 minutes
    agg_df["runs_per90min"] = agg_df.apply(
        lambda row: (row["total"] / row["total_minutes"] * 90) if row["total_minutes"] > 0 else 0, axis=1
    )

    # Pivot the DataFrame to have subtypes as columns and players as rows
    df_pivot = agg_df.pivot(
        index='player_name', 
        columns='event_subtype', 
        values='runs_per90min'
        ).fillna(0)
    
    # Calculate percentiles
    percentile_df = df_pivot.rank(pct=True) * 100

    return df_pivot, percentile_df

def obr_per_position_subtype(
    dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball runs per position groups and event subtype.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns:['position_group', 'event_subtype', 'total', 'targeted', 'untargeted']
    """
    df = off_ball_event_agg(dynamic_events_all, group_by=["event_subtype", "player_position_id"])

    # Map position IDs to position groups
    position_group_map = {"GK": [1], "DEF": [2,3,4,5,6,7,8], "MID": [9,10,11,12,13,14,15], "FWD": [16,17,18,19,20]}
    df["position_group"] = df["player_position_id"].map(lambda x: next((group for group, ids in position_group_map.items() if x in ids), "UNKNOWN"))
    df_grouped = df.groupby(["position_group", "event_subtype"]).agg(
        total=("total", "sum"),
        targeted=("targeted", "sum"),
        untargeted=("untargeted", "sum")
    ).reset_index()

    return df_grouped

def obr_entropy_positiongroup(
    dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the entropy of off-ball runs per position group.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with columns:['position_group', 'entropy']
    """
    df = obr_per_position_subtype(dynamic_events_all)
    df["total_runs"] = (df.groupby("position_group")["total"].transform("sum"))
    df["share"] = df["total"] / df["total_runs"]

    entropy_by_position = (
        df
        .groupby("position_group")["share"]
        .apply(entropy)
        .reset_index(name="entropy")
    )

    # Sort by entropy descending
    entropy_by_position = entropy_by_position.sort_values(by="entropy", ascending=False).reset_index(drop=True)

    return entropy_by_position

def obr_push_defensive_line(
    dynamic_events_all: pd.DataFrame,
    all_metadata: List[Dict[str, Any]],
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0)-> pd.DataFrame:
    """
    Computes statistics on off-ball runs that push the defensive line.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
        all_metadata (List[Dict[str, Any]]): List of match metadata dictionaries.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.

    Returns:
        pd.DataFrame: DataFrame with columns:['player_id', 'total_runs', 'push_defensive_line_runs', 'total_minutes', 'total_runs_per90', 'push_defensive_line_per90', 'percentage_push_defensive_line']
    """

    mid_obr_filtered, eligible_players = filter_eligible_players(
        dynamic_events_all, 
        all_metadata, 
        min_matches, 
        min_avg_minutes_played)
    
    # Filter off-ball runs of subtype 'behind'
    in_behind_runs = mid_obr_filtered.copy()
    in_behind_runs = in_behind_runs[in_behind_runs["event_subtype"] == "behind"]

    # Group by player_id to get total runs and total runs with push_defensive_line equals True
    df_in_behind = (
        in_behind_runs.groupby("player_id", as_index=False)
        .agg(
            total_runs=("event_subtype", "count"),
            push_defensive_line_runs=("push_defensive_line", "sum")
        )
    )

    # Merge to get total minutes played per player
    df_in_behind_merged = df_in_behind.merge(
        eligible_players[["player_id", "total_minutes"]],
        on="player_id",
        how="left"
    )

    # Calculate total runs and push defensive line per 90min
    df_in_behind_merged["total_runs_per90"] = (
        df_in_behind_merged["total_runs"] / df_in_behind_merged["total_minutes"] * 90
    )
    df_in_behind_merged["push_defensive_line_per90"] = (
        df_in_behind_merged["push_defensive_line_runs"] / df_in_behind_merged["total_minutes"] * 90
    )

    # Calculate percentage of runs that push defensive line
    df_in_behind_merged["percentage_push_defensive_line"] = (
        df_in_behind_merged["push_defensive_line_runs"] / df_in_behind_merged["total_runs"] * 100
    )

    return df_in_behind_merged
