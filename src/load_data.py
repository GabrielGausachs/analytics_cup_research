import json
from typing import List, Tuple, Dict, Any
import pandas as pd
import requests
from kloppy.domain import Orientation
from kloppy import skillcorner


def load_matches(matches_json_path: str) -> Tuple[List, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Loads tracking datasets, dynamic events, and metadata for all matches listed in a JSON file.

    This function reads a JSON file containing match IDs, loads:
        - SkillCorner tracking datasets (transformed to static orientation),
        - dynamic event files from GitHub,
        - metadata JSON files for each match.

    Args:
        matches_json_path (str): Path to the local JSON file containing match IDs.
            The JSON is expected to be a list of match dictionaries, each containing an "id" field.

    Returns:
        Tuple[List, pd.DataFrame, List[Dict[str, Any]]]:
            - List of tracking datasets (one per match).
            - A single DataFrame with dynamic events for all matches combined.
            - List of metadata dictionaries (one per match).
    """

    # Load local match list JSON
    with open(matches_json_path, "r") as f:
        matches_json = json.load(f)

    match_ids = [match["id"] for match in matches_json]

    # Load and transform tracking datasets
    all_tracking: List = []
    for match_id in match_ids:
        dataset = skillcorner.load_open_data(
            match_id=match_id,
            coordinates="skillcorner",
            include_empty_frames=False
        )
        dataset.transform(to_orientation=Orientation.STATIC_HOME_AWAY)
        all_tracking.append(dataset)

    # Load dynamic events from GitHub
    all_de_dfs: List[pd.DataFrame] = []
    for match_id in match_ids:
        url = (
            f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/"
            f"matches/{match_id}/{match_id}_dynamic_events.csv"
        )
        try:
            de_match = pd.read_csv(url)
            all_de_dfs.append(de_match)
        except Exception as e:
            print(f"Failed to load dynamic events for match {match_id}: {e}")

    # Combine event DataFrames
    dynamic_events_all = pd.concat(all_de_dfs, ignore_index=True)

    # Load match metadata from GitHub
    all_metadata: List[Dict[str, Any]] = []
    for match_id in match_ids:
        metadata_url = (
            "https://raw.githubusercontent.com/SkillCorner/opendata/"
            "741bdb798b0c1835057e3fa77244c1571a00e4aa/data/matches/"
            f"{match_id}/{match_id}_match.json"
        )
        response = requests.get(metadata_url)
        raw_match_data = response.json()
        all_metadata.append(raw_match_data)

    return all_tracking, dynamic_events_all, all_metadata


        