from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TypedDict

import pandas as pd


@dataclass
class SearchResult:
    """Result of a path search operation.

    Attributes:
        path (List[str]): The list of nodes representing the path from the start to the goal.
        distance (float): The total real-world distance of the path.
        accessibility_cost (float): The total cost (with accessibility considered) of the path.
        nodes_created_count (int): The number of nodes generated during the search.
        nodes_explored_count (int): The number of nodes explored during the search.
    """

    path: List[str]
    distance: float
    accessibility_cost: float
    nodes_created_count: int
    nodes_explored_count: int


@dataclass(order=True)
class Move:
    """Represents a move between nodes with priority for the frontier.

    Attributes:
        source (str): The identifier of the source node.
        destination (str): The identifier of the destination node.
        priority (float): The priority value used by the priority queue (f-score).
        index (int): Tie-breaker index for moves with equal priority.
    """

    source: str = field(compare=False)
    destination: str = field(compare=False)
    priority: float = field(default=0.0)
    index: int = field(compare=False, default=0)


@dataclass
class TestCase:
    """Test case for pathfinding algorithms.

    Attributes:
        adjacency_matrix (pd.DataFrame): Adjacency matrix representing the graph.
        node_coordinates (Dict[str, Tuple[float, float]]): Coordinates of nodes in the graph.
        start (str): Identifier of the starting node.
        goal (str): Identifier of the goal node.
        consider_accessibility (bool): Whether to consider wheelchair accessibility when calculating the cost and heuristic. Defaults to True.
        slope_matrix (Optional[pd.DataFrame]): Slope matrix for accessibility features.
        kerb_ramps_matrix (Optional[pd.DataFrame]): Kerb ramps matrix for accessibility features.
        sidewalk_width_matrix (Optional[pd.DataFrame]): Sidewalk width matrix for accessibility features.

    """

    adjacency_matrix: pd.DataFrame
    node_coordinates: Dict[str, Tuple[float, float]]
    start: str
    goal: str

    consider_accessibility: bool = True
    slope_matrix: Optional[pd.DataFrame] = None
    kerb_ramps_matrix: Optional[pd.DataFrame] = None
    sidewalk_width_matrix: Optional[pd.DataFrame] = None


class BenchmarkResult(TypedDict):
    """Result of a single benchmark test case.

    Attributes:
        algorithm (str): The algorithm used for pathfinding.
        consider_accessibility (bool): Whether wheelchair accessibility was considered.
        start (str): The starting node identifier.
        goal (str): The goal node identifier.
        path (List[str]): The list of nodes representing the found path.
        distance (float): The total real-world distance of the path.
        accessibility_cost (float): The total cost (with accessibility considered) of the path.
        nodes_created_count (int): The number of nodes created during the search.
        nodes_explored_count (int): The number of nodes explored during the search.
        execution_time_ms (float): The time taken to execute the pathfinding algorithm in milliseconds.
    """

    algorithm: str
    consider_accessibility: bool
    start: str
    goal: str
    path: List[str]
    distance: float
    accessibility_cost: float
    nodes_created_count: int
    nodes_explored_count: int
    execution_time_ms: float
