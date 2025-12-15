from .preprocessing import filter_elegible_players
from .tracking_functions import find_frame_start_end, get_player_coordinates, get_opp_team_players_coordinates
from .helpers import get_voronoi_bounded
import numpy as np
from shapely.geometry import box
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from kloppy.domain import TrackingDataset



def space_created(mid_obr: pd.DataFrame, all_tracking: List[TrackingDataset]) -> pd.DataFrame:
    """
    Calculate space created during midfield off-ball runs using Voronoi diagrams. 
    It filters events by duration <= 5 seconds. Calculate the Voronoi area at the 
    start location of the player that does the run in the start and end frame of the event, 
    and computes the difference as space created. 

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

    print("Number of events to process for space created:", len(mid_obr))

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

        # Voronoi at start frame
        poly_start = get_voronoi_bounded(players_start, 0, pitch_bounds)
        area_start = poly_start.area if poly_start else np.nan

        # Voronoi at end frame
        players_end = []
        players_end.append(player_coord)  # same location as start frame
        
        # get all the opponents coordinates
        players_end.extend(get_opp_team_players_coordinates(end_frame, row.team_id))

        players_end = np.array(players_end)

        poly_end = get_voronoi_bounded(players_end, 0, pitch_bounds)
        area_end = poly_end.area if poly_end else np.nan

        # Store in DataFrame
        mid_obr.at[row.Index, 'voronoi_area_start'] = area_start
        mid_obr.at[row.Index, 'voronoi_area_end'] = area_end
        mid_obr.at[row.Index, 'space_created'] = area_end - area_start
        mid_obr.at[row.Index, 'voronoi_poly_start'] = poly_start
        mid_obr.at[row.Index, 'voronoi_poly_end'] = poly_end
    
    print(f"Skipped {i} rows due to missing frames")
    print("Number of events after processing for space created:", len(mid_obr))

    return mid_obr


def metric_sc(
    dynamic_events_all: pd.DataFrame, 
    all_tracking: List[TrackingDataset], 
    all_metadata: List[Dict[str, Any]],
    min_matches: Optional[int] = 0, 
    min_avg_minutes_played: Optional[int] = 0)-> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes space created per 90 minutes for midfielders performing off-ball runs.

    Args:
        dynamic_events_all (pd.DataFrame): DataFrame containing dynamic events data for all matches.
        all_tracking (list): List of tracking datasets for all matches.
        all_metadata (list): List of match metadata dictionaries.
        min_matches (int, optional): Minimum number of matches a player must have played to be included.
        min_avg_minutes_played (int, optional): Minimum average minutes played per match for a player to be included.
    
    Returns:
        tuple: (mid_obr_grouped, mid_obr_merged)
            - mid_obr_grouped (pd.DataFrame): DataFrame grouped by player with total space created, total minutes and space created per 90 minutes.
            - mid_obr_merged (pd.DataFrame): DataFrame with player details and space created metrics.
    """

    # Filter eligible players
    mid_obr_filtered, eligible_players = filter_elegible_players(
        dynamic_events_all, 
        all_metadata,
        min_matches,
        min_avg_minutes_played
    )

    #Calculate space created
    mid_obr_filtered = space_created(mid_obr_filtered, all_tracking)

    # Merge to get total minutes played per player
    mid_obr_merged = mid_obr_filtered.merge(
        eligible_players[["player_id", "total_minutes"]],
        on="player_id",
        how="left"
    )

    # Group by player and calculate total space created
    mid_obr_grouped = mid_obr_merged.groupby("player_id").agg({
        "space_created": "sum",
        "total_minutes": "first"
    }).reset_index()

    # Calculate space created per 90 minutes
    mid_obr_grouped["space_created_per90min"] = (
        mid_obr_grouped["space_created"] / mid_obr_grouped["total_minutes"]
    ) * 90

    # Add space created per 90 min to eligible players dataframe
    mid_obr_merged = eligible_players.merge(
        mid_obr_grouped[["player_id", "space_created_per90min"]],
        on="player_id",
        how="left"
    )

    # filter the dataframe to only specific columns needed for analysis
    columns_needed = ["player_id","player_shortname","third_start","third_end","event_subtype",
                    "voronoi_area_start","voronoi_area_end", "voronoi_poly_start", "voronoi_poly_end",
                    "space_created", "space_created_per90min"]
    
    mid_obr_merged = mid_obr_merged[columns_needed]

    return mid_obr_grouped, mid_obr_merged
