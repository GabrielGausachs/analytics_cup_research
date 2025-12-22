import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import PyPizza, FontManager
from highlight_text import fig_text
import io
import numpy as np
from PIL import Image

font_normal = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
                          'src/hinted/Roboto-Regular.ttf')
font_italic = FontManager('https://raw.githubusercontent.com/googlefonts/roboto/main/'
                          'src/hinted/Roboto-Italic.ttf')
font_bold = FontManager('https://raw.githubusercontent.com/google/fonts/main/apache/robotoslab/'
                        'RobotoSlab[wght].ttf')

subtype_names_map = {
    "behind": "Behind",
    "coming_short": "Coming Short",
    "cross_receiver": "Cross Receiver",
    "dropping_off": "Dropping Off",
    "overlap": "Overlap",
    "pulling_half_space": "Pulling Half-Space",
    "pulling_wide": "Pulling Wide",
    "run_ahead_of_the_ball": "Run Ahead",
    "support": "Support",
    "underlap": "Underlap"
}

metrics_names_map = {
    "ddc_build_up": "DDC",
    "ddc_progression": "DDC",
    "space_created_per90min": "Space Created",
    "xT_progression": "xT",
    "xT_direct": "xT",
    "dist_poss_90": "Distance tip"
}

def radar_plot(
    df_pivot: pd.DataFrame,
    df_percentile: pd.DataFrame,
    team_shortname: str)-> Image.Image:
    """
    Plots a radar chart for a given team showing percentile ranks of off-ball runs per subtype per 90 minutes.

    Args:
        df_pivot (pd.DataFrame): Pivoted DataFrame with off-ball runs per subtype per team normalized per 90 minutes.
                                 Index: team_shortname, Columns: event_subtype
        df_percentile (pd.DataFrame): Percentile DataFrame with off-ball runs per subtype per team normalized per 90 minutes.
                                      Index: team_shortname, Columns: event_subtype
        team_shortname (str): Team shortname to plot.

    Returns:
        Image.Image: The generated radar plot as an image.
    """

    # mapping subtype names
    ordered_subtypes = [
    "cross_receiver",
    "behind",
    "run_ahead_of_the_ball",
    "overlap",
    "underlap",
    "support",
    "coming_short",
    "dropping_off",
    "pulling_half_space",
    "pulling_wide"
    ]

    # Reorder columns 
    df_percentile = df_percentile[ordered_subtypes]
    df_pivot = df_pivot[ordered_subtypes]

    # Extract values
    values = df_percentile.loc[team_shortname].tolist()
    values = [round(v, 1) for v in values]

    params = [subtype_names_map[subtype] for subtype in df_percentile.columns]
    slice_colors = ["#1A78CF"] * 2 + ["#FF9300"] * 4 + ["#D70232"] * 4
    text_colors = ["#000000"] * 10

    baker = PyPizza(
    params=params,                  # list of parameters
    straight_line_color="#F2F2F2",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    other_circle_lw=1,              # linewidth for other circles
    other_circle_ls="-.",           # linestyle for other circles
    inner_circle_size=8,            # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
            values,              # list of values
            figsize=(8, 8),          # adjust figure size
            param_location=110,  # where the parameters will be added
            slice_colors=slice_colors,       # color for individual slices
            value_colors=text_colors,        # color for the value-text
            value_bck_colors=slice_colors,   # color for the blank spaces
            kwargs_slices=dict(
                facecolor="cornflowerblue", edgecolor="black",
                zorder=2, linewidth=1
            ),                   # values to be used when plotting slices
            kwargs_params=dict(
                color="#000000", fontsize=10,
                fontproperties=font_normal.prop, va="center"
            ),                   # values to be used when adding parameter
            kwargs_values=dict(
            color="#000000", fontsize=10,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
                )                   
            )                   # values to be used when adding values            
        )
    
    # add title
    fig_text(
            0.515, 1,
            f"<{team_shortname}> - Runs Profile",
            size=15, fig=fig,
            highlight_textprops=[{"color": "#000000"}],
            ha="center", fontproperties=font_bold.prop, color="#000000"
    )

    fig.patch.set_facecolor("white")    # Figure background
    ax.patch.set_facecolor("white")     # Axes background (if any)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # close the figure to free memory

    return img



def plot_multiple_radar_plots_teams(
    df_pivot: pd.DataFrame,
    df_percentile: pd.DataFrame,
    teams_shortnames: list,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10
    ) -> None:
    """
    Creates a single figure with multiple radar plots next to each other and an insight panel on the right.

    Args:
        df_pivot (pd.DataFrame): Pivoted DataFrame with off-ball runs per subtype per team per 90 minutes.
        df_percentile (pd.DataFrame): Percentile DataFrame with same shape as df_pivot.
        teams_shortnames (list): List of team shortnames to plot.
        season (str): Season name for annotation.
        competition (str): Competition name for annotation.
        total_matches (int): Number of matches for annotation.
    """

    # ----------------------------
    # Left axes: radar plots
    # ----------------------------

    # Generate radar plot images
    radar_images = [radar_plot(df_pivot, df_percentile, team) for team in teams_shortnames]

    n_teams = len(teams_shortnames)


    radar_ratio = 0.85 / n_teams
    widths = [radar_ratio] * n_teams + [0.15]

    fig, axes = plt.subplots(
        1, n_teams + 1, 
        figsize=(18, 8), 
        gridspec_kw={"width_ratios": widths}
    )

    # Ensure axes is always a list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot radar images
    for ax, img in zip(axes[:-1], radar_images):
        ax.imshow(img)
        ax.axis('off')


    # ----------------------------
    # Right axis: text
    # ----------------------------

    ax_text = axes[-1]
    ax_text.axis('off')  # no axes
    ax_text.axvline(x=0, color='black', linewidth=1)  # vertical separator

    # Starting y position (top)
    y_start = 0.95  
    line_spacing = 0.08

    # Line 1: title
    ax_text.text(
        0.05, y_start,
        "INSIGHTS",
        va="top",
        ha="left",
        fontsize=18,
        color='black',
        transform=ax_text.transAxes
    )

    ax_text.text(
        0.05, y_start - line_spacing,
        "Percentile Rank vs A-league teams",
        va="top",
        ha="left",
        fontsize=12,
        color='black',
        transform=ax_text.transAxes,
        wrap=True
    )

    # Define categories and colors
    categories = ["Direct", "Progression", "Build up"]
    colors = ["#1a78cf", "#ff9300", "#d70232"]

    rect_size = 0.05  # size of the square (width = height)
    y_start = y_start - 2 * line_spacing  # adjust starting y position
    line_spacing = 0.08  # space between lines

    for i, (cat, color) in enumerate(zip(categories, colors)):
        y = y_start - i * line_spacing

        # Add square
        ax_text.add_patch(
            plt.Rectangle(
                (0.05, y - rect_size/2),  # center square vertically on the line
                rect_size, rect_size,
                transform=ax_text.transAxes,
                color=color
            )
        )

        # Add text next to square
        ax_text.text(
            0.05 + rect_size + 0.03, y,
            cat,
            va="center",
            ha="left",
            fontsize=12,
            fontproperties=font_bold.prop,
            color="#000000",
            transform=ax_text.transAxes
        )

    # Line 2: main insight
    main_text = "Runtypes define team and player profiles. Understanding these profiles can help optimize tactics and player roles. However, which runs are actually valuable?"

    ax_text.text(
        0.05, y_start - (len(categories)+0.08) * line_spacing,
        main_text,
        va="top",
        ha="left",
        fontsize=12,
        color='black',
        wrap=True,
        transform=ax_text.transAxes
    )

    # Starting y position at the bottom
    y_bottom = 0.02
    line_spacing = 0.03

    # Line 1: Season
    ax_text.text(
        0.05, y_bottom + 2*line_spacing,
        f"Season: {season}",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    # Line 2: Competition
    ax_text.text(
        0.05, y_bottom + line_spacing,
        f"Competition: {competition}",
        va="bottom",
        ha="left",
        fontsize=10, 
        transform=ax_text.transAxes
    )

    # Line 3: Total matches
    ax_text.text(
        0.05, y_bottom,
        f"Total matches: {total_matches}",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    plt.tight_layout()
    plt.show()


def plot_multiple_radar_plots_players(
    df_pivot: pd.DataFrame,
    df_percentile: pd.DataFrame,
    players_names: list,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    min_matches: int = 2,
    min_avg_minutes_played: int = 40
    ) -> None:
    """
    Creates a single figure with multiple radar plots next to each other and an insight panel on the right.

    Args:
        df_pivot (pd.DataFrame): Pivoted DataFrame with off-ball runs per subtype per team per 90 minutes.
        df_percentile (pd.DataFrame): Percentile DataFrame with same shape as df_pivot.
        players_names (list): List of player names to plot.
        season (str): Season name for annotation.
        competition (str): Competition name for annotation.
        total_matches (int): Number of matches for annotation.
        min_matches (int): Minimum number of matches played for annotation.
        min_avg_minutes_played (int): Minimum average minutes played for annotation.
    """

    # ----------------------------
    # Left axes: radar plots
    # ----------------------------

    # Check if the players exist in the DataFrame
    missing_players = [player for player in players_names if player not in df_pivot.index]
    if missing_players:
        raise ValueError(f"The following players are not in the DataFrame index: {missing_players}")
    
    # Generate radar plot images
    radar_images = [radar_plot(df_pivot, df_percentile, player) for player in players_names]

    n_players = len(players_names)


    radar_ratio = 0.85 / n_players
    widths = [radar_ratio] * n_players + [0.15]

    fig, axes = plt.subplots(
        1, n_players + 1, 
        figsize=(18, 8), 
        gridspec_kw={"width_ratios": widths}
    )

    # Ensure axes is always a list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot radar images
    for ax, img in zip(axes[:-1], radar_images):
        ax.imshow(img)
        ax.axis('off')


    # ----------------------------
    # Right axis: text
    # ----------------------------

    ax_text = axes[-1]
    ax_text.axis('off')  # no axes
    ax_text.axvline(x=0, color='black', linewidth=1)  # vertical separator

    # Starting y position (top)
    y_start = 0.95  
    line_spacing = 0.08

    # Line 1: title
    ax_text.text(
        0.05, y_start,
        "INSIGHTS",
        va="top",
        ha="left",
        fontsize=18,
        color='black',
        transform=ax_text.transAxes
    )

    ax_text.text(
        0.05, y_start - line_spacing,
        "Percentile Rank vs A-league midfielders",
        va="top",
        ha="left",
        fontsize=12,
        color='black',
        transform=ax_text.transAxes,
        wrap=True
    )

    # Define categories and colors
    categories = ["Direct", "Progression", "Build up"]
    colors = ["#1a78cf", "#ff9300", "#d70232"]

    rect_size = 0.05  # size of the square (width = height)
    y_start = y_start - 2 * line_spacing  # adjust starting y position
    line_spacing = 0.08  # space between lines

    for i, (cat, color) in enumerate(zip(categories, colors)):
        y = y_start - i * line_spacing

        # Add square
        ax_text.add_patch(
            plt.Rectangle(
                (0.05, y - rect_size/2),  # center square vertically on the line
                rect_size, rect_size,
                transform=ax_text.transAxes,
                color=color
            )
        )

        # Add text next to square
        ax_text.text(
            0.05 + rect_size + 0.03, y,
            cat,
            va="center",
            ha="left",
            fontsize=12,
            fontproperties=font_bold.prop,
            color="#000000",
            transform=ax_text.transAxes
        )

    # Line 2: main insight
    main_text = "Runtypes define team and player profiles. Understanding these profiles can help optimize tactics and player roles. However, which runs are actually valuable?"

    ax_text.text(
        0.05, y_start - (len(categories)+0.08) * line_spacing,
        main_text,
        va="top",
        ha="left",
        fontsize=12,
        color='black',
        wrap=True,
        transform=ax_text.transAxes
    )

    # Starting y position at the bottom
    y_bottom = 0.02
    line_spacing = 0.03

    # Line 1: Season
    ax_text.text(
        0.05, y_bottom + 3*line_spacing,
        f"Season: {season}",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    # Line 2: Competition
    ax_text.text(
        0.05, y_bottom + 2*line_spacing,
        f"Competition: {competition}",
        va="bottom",
        ha="left",
        fontsize=10, 
        transform=ax_text.transAxes
    )

    # Line 3: Total matches
    ax_text.text(
        0.05, y_bottom + line_spacing,
        f"Total matches: {total_matches}",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    # Line 4: Min matches
    ax_text.text(
        0.05, y_bottom,
        f"Minimum of {min_matches} matches and minimum {min_avg_minutes_played} avg minutes played",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    plt.tight_layout()
    plt.show()



def radar_plot_newprofiles(
    df_all: pd.DataFrame,
    player_name: str
    ) -> None:
    """
    Plots a radar chart for overall off-ball runs metrics.

    Args:
        df_all (pd.DataFrame): DataFrame with overall off-ball runs metrics per player normalized per 90 minutes.
                                 Index: player_name, Columns: overall metrics
        player_name (str): Player name to plot.
    """

    # Order of metrics
    ordered_metrics = [
        "ddc_build_up", "space_created_per90min", "ddc_progression",
        "xT_progression", "xT_direct",
        "dist_poss_90"
    ]

    df_pivot = df_all.set_index('player_name')
    df_percentile = df_pivot.rank(pct=True) * 100
    df_percentile = df_percentile[ordered_metrics]
    df_percentile = df_percentile.fillna(50)

    # Extract values
    values = df_percentile.loc[player_name].tolist()
    values = [round(v, 1) for v in values]

    params = [metrics_names_map[metric] for metric in df_percentile.columns]
    slice_colors = ["#D70232"] * 2 + ["#FF9300"] * 2 + ["#1A78CF"] * 1 + ["#008F15"] * 1
    text_colors = ["#000000"] * 6

    baker = PyPizza(
    params=params,                  # list of parameters
    straight_line_color="#F2F2F2",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    other_circle_lw=1,              # linewidth for other circles
    other_circle_ls="-.",           # linestyle for other circles
    inner_circle_size=8,            # size of inner circle
    )

    # plot pizza
    fig, ax = baker.make_pizza(
            values,              # list of values
            figsize=(8, 8),          # adjust figure size
            param_location=110,  # where the parameters will be added
            slice_colors=slice_colors,       # color for individual slices
            value_colors=text_colors,        # color for the value-text
            value_bck_colors=slice_colors,   # color for the blank spaces
            kwargs_slices=dict(
                facecolor="cornflowerblue", edgecolor="black",
                zorder=2, linewidth=1
            ),                   # values to be used when plotting slices
            kwargs_params=dict(
                color="#000000", fontsize=10,
                fontproperties=font_normal.prop, va="center"
            ),                   # values to be used when adding parameter
            kwargs_values=dict(
            color="#000000", fontsize=10,
            fontproperties=font_normal.prop, zorder=3,
            bbox=dict(
                edgecolor="#000000", facecolor="cornflowerblue",
                boxstyle="round,pad=0.2", lw=1
                )                   
            )                   # values to be used when adding values            
        )
    
    # add title
    fig_text(
            0.515, 1,
            f"<{player_name}> - Runs Profile",
            size=15, fig=fig,
            highlight_textprops=[{"color": "#000000"}],
            ha="center", fontproperties=font_bold.prop, color="#000000"
    )

    fig.patch.set_facecolor("white")    # Figure background
    ax.patch.set_facecolor("white")     # Axes background (if any)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)  # close the figure to free memory

    return img
    

def plot_multiple_radar_plots_players_newprofiles(
    df_all: pd.DataFrame,
    players_names: list,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    min_matches: int = 2,
    min_avg_minutes_played: int = 40
    ) -> None:
    """
    Creates a single figure with multiple radar plots next to each other and an insight panel on the right.

    Args:
        df_all (pd.DataFrame): DataFrame with overall off-ball runs metrics per player normalized per 90 minutes.
                                Index: player_name, Columns: overall metrics.
        players_names (list): List of player names to plot.
        season (str): Season name for annotation.
        competition (str): Competition name for annotation.
        total_matches (int): Number of matches for annotation.
        min_matches (int): Minimum number of matches played for annotation.
        min_avg_minutes_played (int): Minimum average minutes played for annotation.
    """

    # ----------------------------
    # Left axes: radar plots
    # ----------------------------

    # Check if the players exist in the DataFrame
    missing_players = [player for player in players_names if player not in df_all["player_name"].values]
    if missing_players:
        raise ValueError(f"The following players are not in the DataFrame index: {missing_players}")
    
    # Generate radar plot images
    radar_images = [radar_plot_newprofiles(df_all, player) for player in players_names]

    n_players = len(players_names)


    radar_ratio = 0.85 / n_players
    widths = [radar_ratio] * n_players + [0.15]

    fig, axes = plt.subplots(
        1, n_players + 1, 
        figsize=(18, 8), 
        gridspec_kw={"width_ratios": widths}
    )

    # Ensure axes is always a list
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    # Plot radar images
    for ax, img in zip(axes[:-1], radar_images):
        ax.imshow(img)
        ax.axis('off')


    # ----------------------------
    # Right axis: text
    # ----------------------------

    ax_text = axes[-1]
    ax_text.axis('off')  # no axes
    ax_text.axvline(x=0, color='black', linewidth=1)  # vertical separator

    # Starting y position (top)
    y_start = 0.95  
    line_spacing = 0.08

    # Line 1: title
    ax_text.text(
        0.05, y_start,
        "INSIGHTS",
        va="top",
        ha="left",
        fontsize=18,
        color='black',
        transform=ax_text.transAxes
    )

    ax_text.text(
        0.05, y_start - line_spacing,
        "Percentile Rank vs A-league midfielders",
        va="top",
        ha="left",
        fontsize=12,
        color='black',
        transform=ax_text.transAxes,
        wrap=True
    )

    # Define categories and colors
    categories = ["Direct", "Progression", "Build up", "Physical"]
    colors = ["#1a78cf", "#ff9300", "#d70232", "#4caf50"]

    rect_size = 0.05  # size of the square (width = height)
    y_start = y_start - 2 * line_spacing  # adjust starting y position
    line_spacing = 0.08  # space between lines

    for i, (cat, color) in enumerate(zip(categories, colors)):
        y = y_start - i * line_spacing

        # Add square
        ax_text.add_patch(
            plt.Rectangle(
                (0.05, y - rect_size/2),  # center square vertically on the line
                rect_size, rect_size,
                transform=ax_text.transAxes,
                color=color
            )
        )

        # Add text next to square
        ax_text.text(
            0.05 + rect_size + 0.03, y,
            cat,
            va="center",
            ha="left",
            fontsize=12,
            fontproperties=font_bold.prop,
            color="#000000",
            transform=ax_text.transAxes
        )

    # Line 2: main insight
    main_text = "Comparison of player off-ball run profiles across key dimensions: Direct, Progression, Build up, and Physical."

    ax_text.text(
        0.05, y_start - (len(categories)+0.08) * line_spacing,
        main_text,
        va="top",
        ha="left",
        fontsize=12,
        color='black',
        wrap=True,
        transform=ax_text.transAxes
    )

    # Starting y position at the bottom
    y_bottom = 0.02
    line_spacing = 0.03

    # Line 1: Season
    ax_text.text(
        0.05, y_bottom + 4*line_spacing,
        f"Season: {season}",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    # Line 2: Competition
    ax_text.text(
        0.05, y_bottom + 3*line_spacing,
        f"Competition: {competition}",
        va="bottom",
        ha="left",
        fontsize=10, 
        transform=ax_text.transAxes
    )

    # Line 3: Total matches
    ax_text.text(
        0.05, y_bottom + 2*line_spacing,
        f"Total matches: {total_matches}",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    # Line 4: Min matches
    ax_text.text(
        0.05, y_bottom + line_spacing,
        f"Minimum of {min_matches} matches and minimum {min_avg_minutes_played} avg minutes played",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    ax_text.text(
        0.05, y_bottom,
        f"Metrics are normalized per 90 minutes played.",
        va="bottom",
        ha="left",
        fontsize=10,
        transform=ax_text.transAxes
    )

    plt.tight_layout()
    plt.show()
