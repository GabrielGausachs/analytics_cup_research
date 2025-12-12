import pandas as pd
from typing import List
from kloppy.domain import TrackingDataset
from .preprocessing import match_minutes_played


def obr_per_subtype(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball event statistics per event subtype.

    This function filters off-ball events (event_type_id == 1), groups them by
    event subtype, and computes:
        - total number of events
        - number of untargeted events (events where `targeted == 0`)

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
            Expected columns include:
                - "event_type_id"
                - "event_subtype"
                - "targeted" (boolean or 0/1 int)

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - "subtype"
            - "total"
            - "untargeted"
    """

    # Filter off-ball events (event_type_id = 1)
    off_ball_events = dynamic_events_all[
        dynamic_events_all["event_type_id"] == 1
    ]

    print("Number of missing 'targeted' values:", off_ball_events['targeted'].isna().sum())

    # Group by subtype and compute stats
    agg_df = (
        off_ball_events
        .groupby("event_subtype")
        .agg(
            total=("targeted", "count"),
            targeted=("targeted", "sum")
        )
        .reset_index()
        .rename(columns={"event_subtype": "subtype"})
    )

    # Compute untargeted events
    agg_df["untargeted"] = agg_df["total"] - agg_df["targeted"]

    return agg_df[["subtype", "total", "untargeted"]]


def obr_per_subtype_per90min(all_tracking: List[TrackingDataset], dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball event statistics per subtype normalized per 90 minutes of play.

    This function calculates total match duration across all tracking datasets,
    gets off-ball event counts per subtype, and normalizes the counts to a 90-minute
    basis. It returns total, targeted, and untargeted counts per 90 minutes.

    Args:
        all_tracking (List[TrackingDataset]): List of tracking datasets for all matches.
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
            Expected columns include:
                - "event_type_id"
                - "event_subtype"
                - "targeted"

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - "subtype"
            - "total_per90min"
            - "targeted_per90min"
            - "untargeted_per90min"
    """

    # Calculate total match duration in minutes
    total_minutes = sum(match_minutes_played(match_tracking) for match_tracking in all_tracking)

    if total_minutes == 0:
        raise ValueError("Total match minutes is zero. Cannot normalize per 90 minutes.")

    # Get off-ball event stats per subtype
    agg_df = obr_per_subtype(dynamic_events_all)

    # Normalize counts per 90 minutes
    agg_df["total_per90min"] = (agg_df["total"] / total_minutes) * 90
    agg_df["targeted_per90min"] = (agg_df["targeted"] / total_minutes) * 90
    agg_df["untargeted_per90min"] = (agg_df["untargeted"] / total_minutes) * 90

    return agg_df[["subtype", "total_per90min", "targeted_per90min", "untargeted_per90min"]]



def obr_per_subtype_per_phase(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the total number of off-ball runs per subtype per phase of play.

    This function filters off-ball events (event_type_id == 1), groups them by
    team_in_possession_phase_type and event_subtype, and counts the number of
    runs.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic event data.
            Expected columns:
                - "event_type_id"
                - "event_subtype"
                - "team_in_possession_phase_type"

    Returns:
        pd.DataFrame: A DataFrame with columns:
            - "phase": The possession phase type.
            - "subtype": Event subtype.
            - "total": Total runs for that subtype and phase.
    """

    # Filter off-ball events
    off_ball_events = dynamic_events_all[dynamic_events_all["event_type_id"] == 1]
    # Group by phase and event subtype
    off_ball_event_groups = off_ball_events.groupby(
        ["team_in_possession_phase_type", "event_subtype"]
    )

    # Compute total run counts
    obr_subtype_phase = [
        {
            "phase": phase,
            "subtype": subtype,
            "total": group.shape[0]
        }
        for (phase, subtype), group in off_ball_event_groups
    ]

    # Convert to DataFrame
    obr_per_subtype_phase = pd.DataFrame(obr_subtype_phase)

    return obr_per_subtype_phase

