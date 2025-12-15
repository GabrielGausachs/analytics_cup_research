import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import PyPizza, add_image, FontManager
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

phase_names_map = {
    "build_up": "Build Up",
    "direct": "Direct",
    "quick_break": "Quick Break",
    "transition": "Transition",
    "create": "Creation",
    "finish": "Finish",
    "chaotic": "Chaotic",
    "disruption": "Disruption",
    "set_play": "Set Play"
}

def plot_total_and_untargeted_per90(
    agg_df: pd.DataFrame,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    title: str = "Off-Ball runs per 90 min by RunType"
    ):
    """
    Plots horizontal bars of total off-ball events per 90 min per subtype,
    overlaying the total of untargeted events with green bars with black edges.
    Adds insights text on the right of the figure.

    Args:
        agg_df (pd.DataFrame): DataFrame returned by `ofb_per_subtype_per90min`.
            Must contain columns:
                - "subtype"
                - "total_per90min"
                - "untargeted_per90min"
        season (str): Season name for annotation.
        competition (str): Competition name for annotation.
        total_matches (int): Number of matches for annotation.
        title (str): Plot title.
    """

    df = agg_df.copy()
    df["subtype_display"] = df["event_subtype"].map(subtype_names_map)
    df = df.sort_values("total_per90min", ascending=True)

    avg_total_per_match = df["total_per90min"].sum()
    avg_untargeted_percent = (df["untargeted_per90min"].sum() / df["total_per90min"].sum()) * 100

    # Create figure with two axes: left for bars, right for text
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, 
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [3, 1]}  # 3:1 ratio
    )

    # ----------------------------
    # Left axis: horizontal bars
    # ----------------------------
    ax = ax_plot
    ax.barh(
        df["subtype_display"],
        df["total_per90min"],
        color="#008F15",
        edgecolor="black",
        linewidth=1.0,
        label="Total per 90 min"
    )

    ax.barh(
        df["subtype_display"],
        df["untargeted_per90min"],
        color="#00ff1e",
        edgecolor="black",
        linewidth=1.5,
        label="Untargeted per 90 min"
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_xlabel("Runs per 90 min", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, df["total_per90min"].max() * 1.1)

    # ----------------------------
    # Right axis: text
    # ----------------------------
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
        fontsize=16,
        color='black',
        transform=ax_text.transAxes
    )

    # Line 2: main insight with bold numbers
    main_text = (
        rf"There are more than $\bf{{{int(avg_total_per_match)}}}$ off-ball runs per match on average, "
        rf"of which $\bf{{{avg_untargeted_percent:.1f}}}\%$ are untargeted. They capture a large part of "
        rf"players’ impact that conventional on-ball metrics miss."
    )

    ax_text.text(
        0.05, y_start - line_spacing,
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
        fontsize=8,
        transform=ax_text.transAxes
    )

    # Line 2: Competition
    ax_text.text(
        0.05, y_bottom + line_spacing,
        f"Competition: {competition}",
        va="bottom",
        ha="left",
        fontsize=8, 
        transform=ax_text.transAxes
    )

    # Line 3: Total matches
    ax_text.text(
        0.05, y_bottom,
        f"Total matches: {total_matches}",
        va="bottom",
        ha="left",
        fontsize=8,
        transform=ax_text.transAxes
    )

    plt.tight_layout()
    plt.show()


def subtype_phase_bubble_plot(
    df_percent: pd.DataFrame,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    title: str = "% Off-ball runs by RunType and Phase"):
    """
    Plots off-ball runs per subtype per phase as a bubble plot. Adds insights text on the right of the figure.

    Args:
        df_percent (pd.DataFrame): Pivoted DataFrame with percentages per subtype per phase.
                                    Index: event_subtype, Columns: phase
        season (str): Season name for annotation.
        competition (str): Competition name for annotation.
        total_matches (int): Number of matches for annotation.
        title (str): Plot title.
    """
    # Map subtype and phase names if provided
    df_plot = df_percent.copy()
    if subtype_names_map:
        df_plot.index = [subtype_names_map.get(idx, idx) for idx in df_plot.index]
    if phase_names_map:
        df_plot.columns = [phase_names_map.get(col, col) for col in df_plot.columns]

    subtypes_ordered = df_plot.index.tolist()
    phases_ordered = df_plot.columns.tolist()

    # Create figure and axis
    fig, (ax_plot, ax_text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [3, 1]})
    ax_plot.set_facecolor("#ffffff")

    # ----------------------------
    # Left axis: bubble plot
    # ----------------------------

    # Plot bubbles
    for i, subtype in enumerate(subtypes_ordered):
        for j, phase in enumerate(phases_ordered):
            pct = df_plot.loc[subtype, phase]
            if pct > 0:
                size = 100 + pct * 18  # bubble size scales with percentage
                alpha = 0.3 + 0.7 * (pct / 100)  # bigger alpha for bigger percentages
                ax_plot.scatter(j, i, s=size, color="#00ff1e", alpha=alpha,
                           edgecolors="black", linewidth=0.8)

                # Add percentage text for noticeable bubbles
                if pct > 10:
                    font_w = 'bold' if pct >= 25 else 'normal'
                    text_size = 5 + pct / 10
                    ax_plot.text(j, i, f"{pct:.0f}%", ha='center', va='center',
                            fontsize=text_size, fontweight=font_w, fontname='Arial')
                    
    ax_plot.set_xticks(range(len(phases_ordered)))
    ax_plot.set_xticklabels(phases_ordered, rotation=45, ha='right')
    ax_plot.set_yticks(range(len(subtypes_ordered)))
    ax_plot.set_yticklabels(subtypes_ordered)


    # Invert y-axis to show first subtype at the top
    ax_plot.invert_yaxis()

    # Spines and grid
    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.spines['bottom'].set_visible(False)
    ax_plot.spines['left'].set_visible(True)
    plt.grid(False)

    # Labels and title
    ax_plot.set_xlabel("Phase", fontsize=12, fontweight='bold')
    ax_plot.set_ylabel("Run type", fontsize=12, fontweight='bold')
    ax_plot.set_title(title, fontsize=14, fontweight='bold', pad = 20)

    # ----------------------------
    # Right axis: insights
    # ----------------------------

    ax_text.axis('off')
    ax_text.axvline(x=0, color='black', linewidth=1)  # vertical separator

    # Insights

    # Starting y position (top)
    y_start = 0.95  
    line_spacing = 0.08

    # Line 1: title
    ax_text.text(
        0.05, y_start,
        "INSIGHTS",
        va="top",
        ha="left",
        fontsize=16,
        color='black',
        transform=ax_text.transAxes
    )

    # Line 2: main insight
    main_text = "Off-ball runs are phase-dependent. Understading their impact in different phases is key."

    ax_text.text(
        0.05, y_start - line_spacing,
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
        fontsize=8,
        transform=ax_text.transAxes
    )

    # Line 2: Competition
    ax_text.text(
        0.05, y_bottom + line_spacing,
        f"Competition: {competition}",
        va="bottom",
        ha="left",
        fontsize=8, 
        transform=ax_text.transAxes
    )

    # Line 3: Total matches
    ax_text.text(
        0.05, y_bottom,
        f"Total matches: {total_matches}",
        va="bottom",
        ha="left",
        fontsize=8,
        transform=ax_text.transAxes
    )


    plt.tight_layout()
    plt.show()


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
    slice_colors = ["#1A78CF"] * 2 + ["#FF9300"] * 3 + ["#D70232"] * 5
    text_colors = ["#000000"] * 5 + ["#F2F2F2"] * 5

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
    
    # add image if badges_dict is provided
    #if badges_dict and team1 in badges_dict:
    #    img = Image.open(badges_dict[team1])
     #   ax_image = add_image(
     #       img, fig, left=0.1, bottom=0.9, width=0.13, height=0.127
     #   )   # these values might differ when you are plotting

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
        "Percentile Rank vs A-league players",
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


def plot_scatter_ddc_distance(
    df_ddc_pd: pd.DataFrame,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    min_matches: int = 2,
    min_avg_minutes_played: int = 40):
    """
    Plots a scatter plot of Defensive Density Change per 90 min vs Distance tip per 90 min.

    Args:
        df_ddc_pd (pd.DataFrame): DataFrame with columns:
            - "def_density_change_per90min"
            - "distance_tip_per90"
        season (str): Season name for annotation.
        competition (str): Competition name for annotation.
        total_matches (int): Number of matches for annotation.
        min_matches (int): Minimum number of matches played for annotation.
        min_avg_minutes_played (int): Minimum average minutes played for annotation.
    """

    # Compute means
    mean_x = df_ddc_pd['def_density_change_per90min'].mean()
    mean_y = df_ddc_pd['distance_tip_per90'].mean()

    # Highest overall player in both metrics
    df_ddc_pd['z_distance'] = (df_ddc_pd['distance_tip_per90'] - mean_y) / df_ddc_pd['distance_tip_per90'].std()
    df_ddc_pd['z_density']  = (df_ddc_pd['def_density_change_per90min'] - mean_x) / df_ddc_pd['def_density_change_per90min'].std()
    df_ddc_pd['overall_score'] = df_ddc_pd['z_distance'] + df_ddc_pd['z_density']
    best_overall_idx = df_ddc_pd.sort_values("overall_score", ascending=False).head(1).index

    # Highest defensive density change and lowest distance tip
    x_max = df_ddc_pd['def_density_change_per90min'].max()
    y_min = df_ddc_pd['distance_tip_per90'].min()
    x_min = df_ddc_pd['def_density_change_per90min'].min()
    y_max = df_ddc_pd['distance_tip_per90'].max()

    # Get the bottom right point
    y_bottom = y_min - 1500
    x_right = x_max + 5

    # Optionally exclude the best overall player
    df_candidates = df_ddc_pd.drop(index=best_overall_idx, errors='ignore')

    x = df_candidates['def_density_change_per90min']
    y = df_candidates['distance_tip_per90']

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    x_right_norm = (x_right - x.min()) / (x.max() - x.min())
    y_bottom_norm = (y_bottom - y.min()) / (y.max() - y.min())

    # Normalized Euclidean distance
    df_candidates['dist_to_bottom_right'] = np.sqrt(
        (x_norm - x_right_norm) ** 2 +
        (y_norm - y_bottom_norm) ** 2
    )

    # Get the 2 closest players
    closest_players = df_candidates.nsmallest(2, 'dist_to_bottom_right')
    best_bottom_right_indices = closest_players.index.tolist()

    
    best_overall_indices = [best_overall_idx[0], best_bottom_right_indices[0], best_bottom_right_indices[1]]

    # Create figure with two axes: left for scatter, right for text
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, 
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [3, 1]}  # 3:1 ratio
    )

    # ----------------------------
    # Left axis: scatter plot
    # ----------------------------
    ax_plot.scatter(df_ddc_pd['def_density_change_per90min'], df_ddc_pd['distance_tip_per90'], 
                    s=200, alpha=0.5, color="#00ff1e", edgecolors='black')

    # Highlight best overall player
    for idx in best_overall_indices:
        ax_plot.scatter(
            df_ddc_pd.loc[idx, 'def_density_change_per90min'],
            df_ddc_pd.loc[idx, 'distance_tip_per90'],
            color='red',
            edgecolor='black',
            s=300,
            alpha=0.8,
            linewidth=1,
        )

        ax_plot.text(
        df_ddc_pd.loc[idx, 'def_density_change_per90min'],
        df_ddc_pd.loc[idx, 'distance_tip_per90'] - 150,
        df_ddc_pd.loc[idx, 'player_short_name'],
        fontsize=10,
        color='black',
        ha='center',
        va='top',
        )

    # Mean lines
    ax_plot.axvline(x=mean_x, color='gray', alpha=0.5, linestyle='--')
    ax_plot.axhline(y=mean_y, color='gray', alpha=0.5, linestyle='--')

    ax_plot.set_xlim(x_min-5, x_max+5)
    ax_plot.set_ylim(y_min-1500, y_max+1500)

    ax_plot.set_xlabel('Defensive Density Change (m)', fontsize=12)
    ax_plot.set_ylabel('Distance tip (m)', fontsize=12)
    ax_plot.set_title('Getting Free from Defensive Pressure', fontsize=14, fontweight='bold')

    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.spines['left'].set_visible(True)
    ax_plot.spines['bottom'].set_visible(True)

    # Axis limits (after plotting everything)
    x_min, x_max = ax_plot.get_xlim()
    y_min, y_max = ax_plot.get_ylim()

    # ----------------------------
    # Quadrant annotations
    # ----------------------------

    # Bottom-right quadrant (farthest corner from mean = bottom-right corner)
    ax_plot.text(
        x_max,
        y_min+100,
        "Efficient movers\n(low volume, high space gain)",
        ha="right",
        va="bottom",
        fontsize=10,
        color="black",
        alpha=0.8
    )

    # Top-right quadrant (farthest corner from mean = top-right corner)
    ax_plot.text(
        x_max,
        y_max-100,
        "Dynamic movers\n(high volume, high space gain)",
        ha="right",
        va="top",
        fontsize=10,
        color="black",
        alpha=0.8
    )


    # ----------------------------
    # Right axis: text
    # ----------------------------

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


    # Line 2: main insight
    main_text = "This highlights players who don’t just cover ground, " \
    "but use their running to regularly get free from pressure and open up space to receive the ball."

    ax_text.text(
        0.05, y_start - line_spacing,
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
        0.05, y_bottom + 5*line_spacing,
        f"Season: {season}",
        va="bottom",
        ha="left",
        fontsize=8,
        transform=ax_text.transAxes
    )

    # Line 2: Competition
    ax_text.text(
        0.05, y_bottom + 4*line_spacing,
        f"Competition: {competition}",
        va="bottom",
        ha="left",
        fontsize=8, 
        transform=ax_text.transAxes
    )

    # Line 3: Total matches
    ax_text.text(
        0.05, y_bottom + 3*line_spacing,
        f"Total matches: {total_matches}",
        va="bottom",
        ha="left",
        fontsize=8,
        transform=ax_text.transAxes
    )

    # Line 4: Min matches
    ax_text.text(
        0.05, y_bottom + 2*line_spacing,
        f"Minimum of {min_matches} matches and minimum {min_avg_minutes_played} avg minutes played",
        va="bottom",
        ha="left",
        fontsize=8,
        transform=ax_text.transAxes
    )

    # Line 5: Metric per90min explanation
    ax_text.text(
        0.05, y_bottom + line_spacing,
        "Metrics are normalized per 90 minutes played.",
        va="bottom",
        ha="left",
        fontsize=8,
        transform=ax_text.transAxes
    )
    
    plt.tight_layout()
    plt.show()
