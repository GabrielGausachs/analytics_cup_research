from .preprocessing import midfielders_obr, player_minutes_per_match
from .tracking_functions import find_frame_start_end, get_player_coordinates, get_opp_team_players_coordinates
import numpy as np
from shapely.geometry import Polygon, box
from scipy.spatial import Voronoi
from typing import List, Dict, Any, Optional
import pandas as pd
from kloppy.domain import TrackingDataset

def get_voronoi_bounded(points, index, pitch_bounds):
    # Add padding points to bound Voronoi
    padding = np.array([
            [-1000, -1000],
            [1000, 1000],
            [1000, -1000],
            [-1000, 1000]
    ])
    points = np.vstack([points, padding])

    # Compute Voronoi
    vor = Voronoi(points)

    # Get region of target player
    region_index = vor.point_region[index]
    region = vor.regions[region_index]

    # Collect finite vertices
    finite_vertices = [vor.vertices[i] for i in region if i != -1]

    poly = Polygon(finite_vertices)

    # Clip with pitch bounds
    poly_clipped = poly.intersection(pitch_bounds)

    if poly_clipped.is_empty:
        raise ValueError("Clipped polygon is empty")

    return poly_clipped

def space_created(mid_obr: pd.DataFrame, all_tracking: List[TrackingDataset]) -> pd.DataFrame:
    """
    Calculate space created metric for off-ball events.

    Args:
        mid_obr (pd.DataFrame): DataFrame of midfield off-ball events.
        all_tracking (list): List of tracking datasets for all matches.
    
    Returns:
        pd.DataFrame: The input DataFrame with an additional columns related to space created,
            representing the space created during the off-ball event.
    """

    # Filter by duration < 5 seconds
    mid_obr = mid_obr[(mid_obr["duration"] <= 5.0)].copy()

    mid_obr = mid_obr.reset_index(drop=True)

    pitch_bounds = box(-52.5, -34, 52.5, 34)  # SkillCorner pitch centered at (0,0)

    # Add a column for Voronoi area changes
    mid_obr["voronoi_area_start"] = np.nan
    mid_obr["voronoi_area_end"] = np.nan
    mid_obr["space_created"] = np.nan
    mid_obr["voronoi_poly_start"] = None
    mid_obr["voronoi_poly_end"] = None

    i = 0
    for row in mid_obr.itertuples():

        # Find start and end frames
        start_frame, end_frame = find_frame_start_end(row, all_tracking)
        if start_frame is None or end_frame is None:
            # remove row from mid_obr to avoid unreliable analysis
            mid_obr.drop(index=row.Index, inplace=True)
            i+=1
            continue

        # Get player index + all positions
        players_start = []

        # Get player coordinates at start frame
        player_coord = get_player_coordinates(start_frame, str(row.player_id))

        if player_coord is None:
            print(f"Player {row.player_id} not found in start frame of match {row.match_id}")
        
        players_start.append(player_coord)
        
        # get all the opponents coordinates
        players_start.extend(get_opp_team_players_coordinates(start_frame, row.team_id))

        players_start = np.array(players_start)

        # 1️⃣ Voronoi at start frame
        poly_start = get_voronoi_bounded(players_start, 0, pitch_bounds)
        area_start = poly_start.area if poly_start else np.nan

        # 2️⃣ Voronoi at end frame but **at the original start location**
        players_end = []
        for idx, (player, coord) in enumerate(start_frame.players_coordinates.items()):
            if player.player_id == str(row.player_id):
                players_end.append(players_start[0])
        
        for idx, (player, coord) in enumerate(end_frame.players_coordinates.items()):
            if team_id != player.team.team_id:
                players_end.append([coord.x, coord.y])

        players_end = np.array(players_end)

        poly_end = get_voronoi_bounded(players_end, 0, pitch_bounds)
        area_end = poly_end.area if poly_end else np.nan

        # Store in DataFrame
        mid_obe.at[row.Index, 'voronoi_area_start'] = area_start
        mid_obe.at[row.Index, 'voronoi_area_end'] = area_end
        mid_obe.at[row.Index, 'space_created'] = area_end - area_start
        mid_obe.at[row.Index, 'voronoi_poly_start'] = poly_start
        mid_obe.at[row.Index, 'voronoi_poly_end'] = poly_end
    
    print(f"Skipped {i} rows due to missing frames")

    return mid_obe


def metric_sc(
    dynamic_events_all: pd.DataFrame, 
    all_tracking: List[TrackingDataset], 
    all_metadata: List[Dict[str, Any]],
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0)-> pd.DataFrame:
    """
    Computes space created per 90 minutes for midfielders performing off-ball runs.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_tracking (list): List of tracking datasets for all matches.
        all_metadata (list): List of match metadata dictionaries.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.
    
    Returns:
        pd.DataFrame: DataFrame with space created per 90 minutes for eligible players.
    """

    mid_obr = midfielders_obr(dynamic_events_all)

    player_minutes_df = player_minutes_per_match(all_metadata)

    # Get eligible players based on min_matches and min_avg_minutes_played
    eligible_players = (
        player_minutes_df.groupby("player_id")
        .agg(
            matches=("match_id", "nunique"),
            avg_minutes=("minutes_played", "mean"),
            total_minutes=("minutes_played", "sum"),
        )
        .reset_index()
    )

    eligible_players = eligible_players[
        (eligible_players["matches"] >= min_matches) &
        (eligible_players["avg_minutes"] >= min_avg_minutes_played)
    ]

    # Filter mid_obr to include only eligible players
    mid_obr_filtered = mid_obr[mid_obr["player_id"].isin(eligible_players["player_id"])]

    #Calculate space created
    mid_obr_filtered = space_created(mid_obr_filtered, all_tracking)

    pass
