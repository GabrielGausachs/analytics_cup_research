import matplotlib.pyplot as plt
import pandas as pd

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
    title: str = "Off-Ball runs per 90 min by Subtype"
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
        rf"of which $\bf{{{avg_untargeted_percent:.1f}}}\%$ are untargeted."
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
    bubble_color: str = "#00ff1e",
    season: str = "2024/2025",
    competition: str = "Australian A-League",
    total_matches: int = 10,
    title: str = "% Off-ball runs by Subtype and Phase"):
    """
    Plots off-ball runs per subtype per phase as a bubble plot. Adds insights text on the right of the figure.

    Args:
        df_percent (pd.DataFrame): Pivoted DataFrame with percentages per subtype per phase.
                                    Index: event_subtype, Columns: phase
        subtype_names_map (dict, optional): Map event_subtype keys to display names
        phase_names_map (dict, optional): Map phase keys to display names
        bubble_color (str): Color of the bubbles
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
                ax_plot.scatter(j, i, s=size, color=bubble_color, alpha=alpha,
                           edgecolors="black", linewidth=0.8)

                # Add percentage text for noticeable bubbles
                if pct > 10:
                    font_w = 'bold' if pct >= 25 else 'normal'
                    text_size = 5 + pct / 10
                    ax_plot.text(j, i, f"{pct:.0f}%", ha='center', va='center',
                            fontsize=text_size, fontweight=font_w, fontname='Arial')
                    
    ax_plot.set_xticks(range(len(phases_ordered)))
    ax_plot.set_xticklabels(phases_ordered, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax_plot.set_yticks(range(len(subtypes_ordered)))
    ax_plot.set_yticklabels(subtypes_ordered, fontsize=12, fontweight='bold')


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

