from typing import List
import pandas as pd

def off_ball_event_agg(dynamic_events_all: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """
    Computes off-ball event statistics grouped by one or more columns.

    Args:
        dynamic_events_all (pd.DataFrame): Dynamic events DataFrame. Must have:
            - 'event_type_id' (off-ball = 1)
            - 'targeted' (boolean)
        group_by (List[str]): List of columns to group by, e.g.
            ['event_subtype'] or ['team_in_possession_phase_type', 'event_subtype']

    Returns:
        pd.DataFrame: Aggregated DataFrame with columns:
            - all group_by columns
            - 'total': total events in group
            - 'targeted': number of targeted events
            - 'untargeted': number of untargeted events
    """

    # Filter off-ball events
    off_ball_events = dynamic_events_all[dynamic_events_all["event_type_id"] == 1]

    # Aggregate counts
    agg_df = (
        off_ball_events
        .groupby(group_by)
        .agg(
            total=("targeted", "count"),
            targeted=("targeted", "sum")
        )
        .reset_index()
    )

    # Compute untargeted
    agg_df["untargeted"] = agg_df["total"] - agg_df["targeted"]

    return agg_df


def normalize_per90min(df: pd.DataFrame, total_minutes: float, columns_to_normalize: List[str]) -> pd.DataFrame:
    """
    Normalizes specified count columns per 90 minutes of play.

    Args:
        df (pd.DataFrame): DataFrame containing counts to normalize.
        total_minutes (float): Total match duration in minutes.
        columns_to_normalize (List[str]): List of column names to normalize per 90 min.

    Returns:
        pd.DataFrame: Copy of the DataFrame with normalized columns.
            Each normalized column is named '<original_column>_per90min'.
    """

    if total_minutes == 0:
        raise ValueError("Total match minutes is zero. Cannot normalize per 90 minutes.")

    df_norm = df.copy()
    for col in columns_to_normalize:
        if col not in df_norm.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        df_norm[f"{col}_per90min"] = (df_norm[col] / total_minutes) * 90

    return df_norm



