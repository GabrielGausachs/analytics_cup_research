from shapely import Polygon
import numpy as np
from shapely.geometry import box
from typing import List
import pandas as pd
from kloppy.domain import TrackingDataset
from mplsoccer import Pitch
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from IPython.display import HTML

from ..utils.tracking_functions import get_player_coordinates, get_opp_team_players_coordinates, get_frame_object, get_team_players_coordinates
from ..utils.helpers import get_voronoi_bounded


def animate_space_created(
    event_row: pd.Series,
    all_tracking: List[TrackingDataset],
    interval: int = 100  # ms between frames
    ) -> FuncAnimation:
    """
    Animate off-ball run and surrounding players.
    """

    # --- Draw pitch once ---
    pitch = Pitch(
        pitch_type="skillcorner",
        pitch_length=105,
        pitch_width=68,
        pitch_color="#001400",
        line_color="white",
        linewidth=1.5
    )
    fig, ax = pitch.draw(figsize=(10, 6))

    # --- Info text ---
    info_text = (
        f"Player: {event_row.player_name}\n"
        f"Team: {event_row.team_shortname}\n"
        f"Run type: {event_row.event_subtype}\n"
        f"Avg space created: {event_row.space_created:.2f} m²"
    )

    info_box = ax.text(
        -51, 33,
        info_text,
        fontsize=10,
        color="white",
        ha="left",
        va="top",
        zorder=20
    )

    info_box.set_animated(True)

    first_frame = int(event_row.frame_start)
    last_frame = int(event_row.end_frame_sc)
    frames = list(range(first_frame, last_frame + 1))

    # --- Team colors ---
    team_colors = {
        "team": "#1A78CF",
        "opponent": "#D70232",
        "off_ball_runner": "#00FF00",
        "ball": "#FFD700",
    }

    first_frame_object = get_frame_object(int(event_row.match_id), first_frame, all_tracking)
    runner_starting_point = get_player_coordinates(first_frame_object, event_row.player_id)
    

    # --- Initialize empty scatter artists ---
    teammates_scatter = ax.scatter([], [], s=200, c=team_colors["team"],
                                   edgecolors="white", linewidths=1.5, zorder=9)

    opponents_scatter = ax.scatter([], [], s=200, c=team_colors["opponent"],
                                   edgecolors="white", linewidths=1.5, zorder=9)

    ball_scatter = ax.scatter([], [], s=100, c=team_colors["ball"],
                              edgecolors="white", linewidths=1.5, zorder=9)
    
    runner_scatter = ax.scatter([], [], s=200, c=team_colors["off_ball_runner"],
                                edgecolors="white", linewidths=1.5, zorder=9)

    # --- Voronoi patch container ---
    voronoi_patch = [None]  # mutable so we can replace it

    # --- Update function ---
    def update(frame_id):
        frame = get_frame_object(int(event_row.match_id), frame_id, all_tracking)
        if frame is None:
            return runner_scatter, teammates_scatter, opponents_scatter, ball_scatter, info_box

        # Player
        runner = get_player_coordinates(frame, event_row.player_id)
        runner_scatter.set_offsets([runner])

        # Teammates
        mates = get_team_players_coordinates(frame, event_row.team_id)
        teammates_scatter.set_offsets(np.array(mates) if mates else np.empty((0, 2)))

        # Opponents
        opps = get_opp_team_players_coordinates(frame, event_row.team_id)
        opponents_scatter.set_offsets(np.array(opps) if opps else np.empty((0, 2)))

        # Ball
        ball = (frame.ball_coordinates.x, frame.ball_coordinates.y)
        ball_scatter.set_offsets([ball])

        # --- Voronoi calculation ---
        all_players = [runner_starting_point]
        all_players.extend(mates if mates else [])
        all_players.extend(opps if opps else [])
        all_players = np.array(all_players)

        pitch_bounds = box(-52.5, -34, 52.5, 34)  # SkillCorner pitch centered at (0,0)
        voronoi_poly = get_voronoi_bounded(all_players, 0, pitch_bounds)

        # Remove old Voronoi patch
        if voronoi_patch[0] is not None:
            voronoi_patch[0].remove()

        # Draw new Voronoi polygon with fill
        if voronoi_poly and not voronoi_poly.is_empty and isinstance(voronoi_poly, Polygon):
            x, y = voronoi_poly.exterior.xy
            patch = ax.fill(x, y, facecolor="yellow", alpha=0.3, edgecolor="yellow", lw=2, zorder=5)[0]
            voronoi_patch[0] = patch
        else:
            voronoi_patch[0] = None
            
        return runner_scatter, teammates_scatter, opponents_scatter, ball_scatter, voronoi_patch[0], info_box

    # --- Create animation ---
    anim = FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=interval,
        blit=True
    )

    return anim


def animate_run_by_event_id(
    event_id: str,
    df_runs: pd.DataFrame,
    all_tracking: List[TrackingDataset],
    interval: int = 200,
    display: bool = True) -> FuncAnimation:
    """
    Animate a single off-ball run identified by event_id.

    Args:
        event_id (str): The event ID of the off-ball run to animate.
        df_runs (pd.DataFrame): DataFrame containing off-ball run events.
        all_tracking (list): List of tracking datasets for all matches.
        interval (int, optional): Time in milliseconds between frames in the animation. Defaults to 100.
        display (bool, optional): Whether to display the animation in a Jupyter notebook. Defaults to True.
    
    Returns:
        FuncAnimation: The animation object.
    """

    # --- Select the run ---
    run_row = df_runs.loc[df_runs["event_id"] == event_id]

    if run_row.empty:
        raise ValueError(f"No run found with event_id = {event_id}")

    if len(run_row) > 1:
        raise ValueError(f"Multiple runs found with event_id = {event_id}")

    event_row = run_row.iloc[0]

    # --- Create animation ---
    anim = animate_space_created(
        event_row=event_row,
        all_tracking=all_tracking,
        interval=interval
    )

     # --- Close the figure so it doesn't show a static last frame ---
    anim._fig.tight_layout()   # optional: tidy layout
    plt.close(anim._fig)
    # --- Display in notebook ---
    if display:
        return HTML(anim.to_jshtml())

    return anim