import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from typing import Tuple

def entropy(x: np.ndarray) -> float:
    """
    Compute the Shannon entropy of a probability distribution.

    Zero-valued entries are ignored to avoid numerical issues with log(0).

    Args:
        x (np.ndarray): 1D array representing a probability distribution.
            Values are assumed to be non-negative and sum to 1.

    Returns:
        float: Shannon entropy of the distribution.
    """
    x = x[x > 0]
    return float(-(x * np.log(x)).sum())


def get_voronoi_bounded(
    points: np.ndarray,
    index: int,
    pitch_bounds: Polygon
    ) -> Polygon:
    """
    Compute a bounded Voronoi cell for a single point and clip it to the pitch.

    This function constructs a Voronoi diagram from a set of 2D points
    (e.g. player positions), extracts the Voronoi region corresponding
    to a specific point, and clips it to the provided pitch boundaries.
    Artificial padding points are added to ensure the Voronoi regions
    are finite.

    Args:
        points (np.ndarray): Array of shape (N, 2) containing the players coordinates
        index (int): Index of the player for which the Voronoi
            region is computed.
        pitch_bounds (Polygon): Shapely polygon representing the skill corner pitch
            boundaries used to clip the Voronoi region.

    Returns:
        Polygon: A Shapely polygon representing the bounded Voronoi cell
        of the selected point, clipped to the pitch.
    """
    
    # Add padding points to bound Voronoi
    padding = np.array([
        [-1000.0, -1000.0],
        [1000.0,  1000.0],
        [1000.0, -1000.0],
        [-1000.0,  1000.0]
    ])
    points = np.vstack([points, padding])

    # Compute Voronoi diagram
    vor = Voronoi(points)

    # Get region of target point
    region_index = vor.point_region[index]
    region = vor.regions[region_index]

    # Collect finite vertices (exclude infinite ones marked as -1)
    finite_vertices = [vor.vertices[i] for i in region if i != -1]

    poly = Polygon(finite_vertices)

    # Clip with pitch bounds
    poly_clipped = poly.intersection(pitch_bounds)

    if poly_clipped.is_empty:
        raise ValueError("Clipped polygon is empty")

    return poly_clipped

def whisker_bounds_numpy(x: np.ndarray) -> Tuple[float, float]:
    """
    Compute the lower and upper whisker bounds for outlier detection using the IQR method.

    Args:
        x (np.ndarray): 1D array of numeric values.

    Returns:
        Tuple[float, float]: Lower and upper whisker bounds.
    """
    x = x[~np.isnan(x)]
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr
