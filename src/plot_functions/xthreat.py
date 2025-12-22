import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.helpers import annotate_above_below

def plot_violin_xthreat(
    df_xthreat_zscore: pd.DataFrame,
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    min_matches: int = 2,
    min_avg_minutes_played: int = 40) -> None:
    """
    Plots a violin plot of standardized xThreat per 90 min by run groups (Progression, Direct).

    Args:
        df_xthreat_zscore (pd.DataFrame): DataFrame with columns:
            - "run_group": run group of the off-ball run
            - "xthreat_z": z-score of xThreat per 90 within that run group
            - "total_runs": total number of runs ending in that run group
        season (str): Season name for annotation.
        competition (str): Competition name for annotation.
        total_matches (int): Number of matches for annotation.
        min_matches (int): Minimum number of matches played for annotation.
        min_avg_minutes_played (int): Minimum average minutes played for annotation.
    """

    # Identify top 2 players in each run group
    top_players = (
        df_xthreat_zscore.sort_values("z_score", ascending=False)
        .groupby("run_group")
        .head(2))

    # Create figure with two axes: left for scatter, right for text
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2, 
        figsize=(12, 6),
        gridspec_kw={"width_ratios": [3, 1]}  # 3:1 ratio
    )

    # ----------------------------
    # Left axis: violin plot
    # ----------------------------
    
    # Violin (distribution)
    sns.violinplot(
        data=df_xthreat_zscore,
        y="run_group",
        x="z_score",
        inner=None,
        linewidth=1,
        color="#00ff1e",
        alpha=0.5,
        ax=ax_plot
    )

    # Scatter with size mapped to number of runs
    sns.scatterplot(
        data=df_xthreat_zscore,
        y="run_group",
        x="z_score",
        size="total_runs",
        sizes=(30, 250),          # controls min/max dot size
        alpha=1,
        color="#008F15",
        legend="brief",
        ax=ax_plot
    )

    # Highlight top players
    sns.scatterplot(
        data=top_players,
        y="run_group",
        x="z_score",
        size="total_runs",
        sizes=(50, 280),
        color="red",
        edgecolor="black",
        linewidth=1,
        legend=False,
        ax=ax_plot
    )

    ax_plot.set_xlabel("Standardised xThreat per 90 min", fontsize=12, fontweight='bold')
    ax_plot.set_ylabel("Run Group", fontsize=12, fontweight='bold')
    ax_plot.set_title("Off-ball Threat Creation by Midfielders by Run Group", fontsize=14, fontweight='bold')

    ax_plot.spines['top'].set_visible(False)
    ax_plot.spines['right'].set_visible(False)
    ax_plot.spines['left'].set_visible(True)
    ax_plot.spines['bottom'].set_visible(True)

    handles, labels = ax_plot.get_legend_handles_labels()
    ax_plot.legend(
        handles=handles,
        labels=labels,
        title="Number of runs",
        loc="upper right",
        frameon=False
    )

    third_order = ["Direct", "Progression"] 
    y_pos = {v: i for i, v in enumerate(third_order)}

    # Keep track of how many top players we've annotated per group
    group_counter = {"Direct": 0, "Progression": 0}

    for _, row in top_players.iterrows():
        group = row["run_group"]
        y = y_pos[group]

        # Alternate direction: even index = above, odd index = below
        direction = "above" if group_counter[group] % 2 == 0 else "below"

        annotate_above_below(
            ax=ax_plot,
            x=row["z_score"],
            y=y,
            label=row["player_name"],
            direction=direction
        )

        group_counter[group] += 1



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
    main_text = "xThreat measures the probability of a goal being scored within 10 seconds of a pass." \
    "Values are standardised within each run group, showing which players generate above-average threat from off-ball runs in an specific run types."

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
    # Line 2: main insight
    main_text = "This plot reveals midfielders whose off-ball runs turn movement into measurable threat, " \
    "helping to identify the most impactful contributors in each run group."

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
    
    sns.despine(left=True)
    plt.tight_layout()
    plt.show()