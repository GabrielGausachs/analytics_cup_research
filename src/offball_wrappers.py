import pandas as pd
from kloppy.domain import TrackingDataset
from .preprocessing import match_minutes_played
from typing import List
from .aggregates import off_ball_event_agg, normalize_per90min


def obr_per_subtype(dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball event statistics per event subtype.
    Wrapper around off_ball_event_agg.
    """
    return off_ball_event_agg(dynamic_events_all, group_by=["event_subtype"])


def obr_per_subtype_per90min(all_tracking: List[TrackingDataset], dynamic_events_all: pd.DataFrame) -> pd.DataFrame:
    """
    Computes off-ball event stats per subtype normalized per 90 minutes of play.

    Uses off_ball_event_agg and normalize_per90min.
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

    Uses the flexible aggregation function.
    """
    return off_ball_event_agg(
        dynamic_events_all,
        group_by=["team_in_possession_phase_type", "event_subtype"]
    )[["team_in_possession_phase_type", "event_subtype", "total"]]