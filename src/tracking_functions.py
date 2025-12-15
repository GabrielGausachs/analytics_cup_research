from kloppy.domain import TrackingDataset
import pandas as pd
from typing import List, Dict, Any, Tuple

def tracking_index(all_tracking: List[TrackingDataset]) -> Dict[int, Dict[int, Any]]:
    """
    Builds fast lookup dictionaries for matches and frames.

    Args:
        all_tracking (list): List of tracking datasets for all matches.

    Returns:
        dict: A dictionary where keys are match IDs and values are
            dictionaries mapping frame IDs to frame objects.
    """
    tracking_index = {}

    for match in all_tracking:
        match_id = int(match.metadata.game_id)
        tracking_index[match_id] = {
            int(frame.frame_id): frame for frame in match.frames
        }

    return tracking_index


def find_frame_start_end(row: pd.Series, all_tracking: List[TrackingDataset]) -> Tuple[Any, Any]:
    """
    Find start and end frames objects from the tracking data for a given off-ball event row.

    Args:
        row (pd.Series): A row from the dynamic_events_all DataFrame.
        all_tracking (list): List of tracking datasets for all matches.

    Returns:
        tuple: (start_frame, end_frame) if found, else (None, None).
    """

    match_id = int(row.match_id)
    tracking_idx = tracking_index(all_tracking)

    match_frames = tracking_idx.get(match_id)
    if match_frames is None:
        raise ValueError(f"Match {match_id} not found in tracking_index")

    start_frame = match_frames.get(int(row.frame_start))
    end_frame   = match_frames.get(int(row.frame_end))

    return start_frame, end_frame

def get_player_coordinates(frame: Any, player_id: str) -> Tuple[float, float]:
    """
    Get the (x, y) coordinates of a player in a given frame.

    Args:
        frame (Any): A frame object from the tracking data.
        player_id (str): The ID of the player.

    Returns:
        tuple: (x, y) coordinates of the player if found, else (None, None).
    """
    for player, coord in frame.players_coordinates.items():
        if player.player_id == player_id:
            return coord.x, coord.y
    return None, None

def get_opp_team_players_coordinates(frame: Any, team_id: int) -> List[Tuple[float, float]]:
    """
    Get the (x, y) coordinates of all players from the opposing team in a given frame.

    Args:
        frame (Any): A frame object from the tracking data.
        team_id (int): The ID of not the opposing team.

    Returns:
        list: List of (x, y) coordinates of the opposing team's players.
    """
    coordinates = []
    for player, coord in frame.players_coordinates.items():
        if player.team.team_id != team_id:
            coordinates.append((coord.x, coord.y))
    return coordinates