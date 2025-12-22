import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from ..utils.preprocessing import filter_eligible_players


def obr_xthreat(
    dynamic_events_all: pd.DataFrame,
    all_metadata: List[Dict[str, Any]],
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0) -> pd.DataFrame:
    """
    Computes xThreat contribution, total runs, xpass completion, and number of dangerous not difficult runs 
    per 90 minutes for midfielders runs. Also provides per-rungroup breakdowns.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame.
        all_metadata (List[Dict[str, Any]]): List of match metadata dictionaries.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.

    Returns: 
        tuple: (df_player, df_rungroups)
            - df_player (pd.DataFrame): DataFrame grouped by player.
            - df_rungroups (pd.DataFrame): DataFrame grouped by player and run group.

    """

    
    mid_obr_filtered, eligible_players = filter_eligible_players(
        dynamic_events_all, 
        all_metadata, 
        min_matches, 
        min_avg_minutes_played
    )

    # Define run groups
    progression_runs = [
        "run_ahead_of_the_ball", "overlap", "underlap", "support"]

    direct_runs = [
        "cross_receiver", "behind"]

    valid_subtypes = progression_runs + direct_runs

    p_d_runs = mid_obr_filtered[
        mid_obr_filtered["event_subtype"].isin(valid_subtypes)].copy()
    
    # Assign run group
    p_d_runs["run_group"] = np.where(
        p_d_runs["event_subtype"].isin(progression_runs),
        "Progression",
        "Direct"
    )

    # Calculate total metrics per player
    df_player = (
        p_d_runs.groupby("player_id", as_index=False)
        .agg(
            total_runs=("event_subtype", "count"),
            xthreat=("xthreat", "sum"),
            xpass_completion=("xpass_completion", "sum"),
            dangerous_not_difficult=(
                "event_subtype",
                lambda x: (
                    (p_d_runs.loc[x.index, "dangerous"]) &
                    (~p_d_runs.loc[x.index, "difficult_pass_target"])
                    ).sum()
                )
            )
    )

    # Calculate per-player per-rungroup metrics
    df_rungroups = (
        p_d_runs.groupby(["player_id", "run_group"], as_index=False)
        .agg(
            total_runs=("event_subtype", "count"),
            xthreat=("xthreat", "sum"),
            xpass_completion=("xpass_completion", "sum"),
            dangerous_not_difficult=(
                "event_subtype",
                lambda x: (
                    (p_d_runs.loc[x.index, "dangerous"]) &
                    (~p_d_runs.loc[x.index, "difficult_pass_target"])
                    ).sum()
                )
            )
    )

    # Merge to get total minutes
    df_player = df_player.merge(
        eligible_players[["player_id", "total_minutes"]],
        on="player_id",
        how="left"
    )

    df_rungroups = df_rungroups.merge(
        eligible_players[["player_id", "total_minutes"]],
        on="player_id",
        how="left"
    )


    # Create per90 metrics
    per90_cols = [
        "total_runs",
        "xthreat",
        "xpass_completion",
        "dangerous_not_difficult"
    ]

    for col in per90_cols:
        df_rungroups[f"{col}_per90"] = df_rungroups[col] / df_rungroups["total_minutes"] * 90
        df_player[f"{col}_per90"] = df_player[col] / df_player["total_minutes"] * 90

    # Add player_name to both dataframes from dynamic_events_all
    player_names = (
        dynamic_events_all[["player_id", "player_name"]]
        .drop_duplicates()
    )
    df_player = df_player.merge(
        player_names,
        on="player_id",
        how="left"
    )
    df_rungroups = df_rungroups.merge(
        player_names,
        on="player_id",
        how="left"
    )
    
    return df_player, df_rungroups