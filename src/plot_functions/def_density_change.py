import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_scatter_ddc_distance(
    df_ddc_pd: pd.DataFrame,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    min_matches: int = 2,
    min_avg_minutes_played: int = 40) -> None:
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
        df_ddc_pd.loc[idx, 'distance_tip_per90'] - 175,
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

    ax_plot.set_xlabel('Defensive Density Change (m)', fontsize=12, fontweight='bold')
    ax_plot.set_ylabel('Distance tip (m)', fontsize=12, fontweight='bold')
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

    # Line 2: description metric
    main_text = "This compares how the average distance to nearby opponents (within 10 m) around a player " \
    "changes from the start to the end of an off-ball run, alongside the distance covered while the team is in possession."

    ax_text.text(
        0.05, y_start - line_spacing,
        main_text,
        va="top",
        ha="left",
        fontsize=10,
        color='black',
        wrap=True,
        transform=ax_text.transAxes
    )

    # Line 3: main insight
    main_text = "Highlighted players don’t just cover ground, " \
    "but use their running to regularly get free from pressure and open up space to receive the ball."

    ax_text.text(
        0.05, y_start - 3.25*line_spacing,
        main_text,
        va="top",
        ha="left",
        fontsize=10,
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