from datetime import timedelta
from kloppy.domain import TrackingDataset
import pandas as pd
from typing import List, Dict, Any


def match_minutes_played(match_tracking: TrackingDataset) -> float:
    """
    Computes the total number of minutes played in a match based on the
    tracking dataset's period timestamps.

    The function iterates over all periods in the match metadata and sums
    their durations. It then returns the total played time in minutes.

    Args:
        match_tracking (TrackingDataset): A kloppy tracking dataset.
            It must contain `metadata.periods`, where each period has
            `start_timestamp` and `end_timestamp` attributes.

    Returns:
        float: Total match duration in minutes.
    """

    total_duration = timedelta(0)

    for period in match_tracking.metadata.periods:
        period_duration = period.end_timestamp - period.start_timestamp
        total_duration += period_duration

    total_minutes = total_duration.total_seconds() / 60

    return total_minutes


def player_minutes_per_match(all_metadata: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Return minutes played per player per match.

    Args:
        all_metadata (list): List of match metadata dictionaries.

    Returns:
        pd.DataFrame: DataFrame with columns
            ['match_id', 'player_id', 'minutes_played']
    """

    records = []

    for metadata in all_metadata:
        match_id = metadata.get("id")

        for player in metadata.get("players", []):
            player_id = player.get("id")

            if player_id is None:
                continue

            total_time = player.get("playing_time", {}).get("total", {})

            if total_time is None:
                continue
            else:
                minutes = total_time.get("minutes_played", 0)

            records.append({
                "match_id": match_id,
                "player_id": player_id,
                "minutes_played": minutes
            })

    return pd.DataFrame(records)

