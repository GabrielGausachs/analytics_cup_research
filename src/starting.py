from kloppy import skillcorner
import os
import json

matches_json_path = os.path.join(os.path.dirname(__file__), "data/matches.json")

with open(matches_json_path, "r") as f:
    matches_json = json.load(f)

match_id = matches_json[0]["id"]

tracking_data_github_url = f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
meta_data_github_url = f"https://raw.githubusercontent.com/SkillCorner/opendata/master/data/matches/{match_id}/{match_id}_match.json"

dataset = skillcorner.load(
    meta_data=meta_data_github_url,
    raw_data=tracking_data_github_url,
    # Optional Parameters
    coordinates="skillcorner",  # or specify a different coordinate system
    sample_rate=(1 / 2),  # changes the data from 10fps to 5fps
    limit=100,  # only load the first 100 frames
)

df = (
    dataset.transform(
        to_orientation="STATIC_HOME_AWAY"
    )  # Now, all attacks happen from left to right
    .to_df(
        engine="pandas"
    )  # Convert to a Polars DataFrame, or use engine="pandas" for a Pandas DataFrame
)