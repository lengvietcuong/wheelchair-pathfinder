from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, TypedDict

import pandas as pd


class SearchResult(NamedTuple):
    """Result of a path search operation.

    Attributes:
        path (List[str]): The list of nodes representing the path from the start to the goal.
        cost (float): The total cost associated with the found path.
        nodes_created_count (int): The number of nodes generated during the search.
    """

    path: List[str]
    cost: float
    nodes_created_count: int


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


class Algorithm(Enum):
    """Enum of available pathfinding algorithms."""

    A_STAR = "A*"
    DFS = "Depth-First Search (DFS)"
    BFS = "Breadth-First Search (BFS)"
    DIJKSTRA = "Dijkstra"


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
        path_cost (float): The total cost of the found path.
        path_length (int): The length of the found path.
        nodes_created_count (int): The number of nodes created during the search.
        execution_time_ms (float): The time taken to execute the pathfinding algorithm in milliseconds.
    """

    algorithm: str
    consider_accessibility: bool
    start: str
    goal: str
    path: List[str]
    path_cost: float
    path_length: int
    nodes_created_count: int
    execution_time_ms: float
