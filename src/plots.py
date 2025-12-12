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

def plot_total_and_untargeted_per90(agg_df: pd.DataFrame, title: str = "Off-Ball Events per 90 min"):
    """
    Plots horizontal bars of total off-ball events per 90 min per subtype,
    overlaying the proportion of untargeted events with green bars with black edges.

    Args:
        agg_df (pd.DataFrame): DataFrame returned by `ofb_per_subtype_per90min`.
            Must contain columns:
                - "subtype"
                - "total_per90min"
                - "untargeted_per90min"
        title (str): Plot title.
    """

    # Compute proportion of untargeted per total
    agg_df = agg_df.copy()
    agg_df["untargeted_proportion"] = agg_df["untargeted_per90min"] / agg_df["total_per90min"]
    agg_df["subtype_display"] = agg_df["subtype"].map(subtype_names_map)

    # Sort by total events for better visualization
    agg_df = agg_df.sort_values("total_per90min", ascending=True)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Plot total events per 90 min as base bars (light gray)
    ax.barh(
        agg_df["subtype_display"],
        agg_df["total_per90min"],
        color="#d3d3d3",
        edgecolor="black",
        linewidth=1.0,
        label="Total per 90 min"
    )

    # Overlay untargeted proportion as green bars
    ax.barh(
        agg_df["subtype_display"],
        agg_df["total_per90min"] * agg_df["untargeted_proportion"],
        color="#00ff1e",
        edgecolor="black",
        linewidth=1.5,
        label="% Untargeted"
    )

    # Aesthetics: remove top/right spines, keep left/bottom
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Remove grid
    plt.grid(False)

    # Y-axis labels
    plt.yticks(fontsize=12, fontweight='normal', fontname='Arial')

    # X-axis label
    plt.xlabel("Events per 90 min", fontsize=12, fontweight='bold')

    # Title
    plt.title(title, fontsize=14, fontweight='bold')

    # Legend
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
