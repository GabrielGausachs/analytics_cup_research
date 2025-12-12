from datetime import timedelta
from kloppy.domain import TrackingDataset


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


