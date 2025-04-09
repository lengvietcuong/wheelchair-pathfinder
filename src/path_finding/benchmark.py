import itertools
import logging
import random
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .a_star import AStar
from .bfs import BFS
from .custom_types import (
    Algorithm,
    BenchmarkResult,
    TestCase,
)
from .dfs import DFS
from .dijkstra import Dijkstra
from .path_finder import PathFinder


PATH_FINDERS: Dict[Algorithm, PathFinder] = {
    Algorithm.A_STAR: AStar,
    Algorithm.DFS: DFS,
    Algorithm.BFS: BFS,
    Algorithm.DIJKSTRA: Dijkstra,
}


logger = logging.getLogger(__name__)


def run_benchmark(
    algorithm: Algorithm,
    test_case: TestCase,
) -> BenchmarkResult:
    """
    Run a benchmark for the given pathfinder and test case.

    Args:
        algorithm (Algorithm): Pathfinding algorithm to use.
        test_case (TestCase): Test case configurations.

    Returns:
        BenchmarkResult: The result of running the benchmark.
    """
    pathfinder: PathFinder = PATH_FINDERS[algorithm](
        adjacency_matrix=test_case.adjacency_matrix,
        node_coordinates=test_case.node_coordinates,
        slope_matrix=test_case.slope_matrix,
        kerb_ramps_matrix=test_case.kerb_ramps_matrix,
        sidewalk_width_matrix=test_case.sidewalk_width_matrix,
    )

    start_time = time.perf_counter()
    search_result = pathfinder.find_path(
        test_case.start, test_case.goal, test_case.consider_accessibility
    )
    execution_time_ms = (time.perf_counter() - start_time) * 1_000

    return BenchmarkResult(
        algorithm=algorithm.value,
        consider_accessibility=test_case.consider_accessibility,
        start=test_case.start,
        goal=test_case.goal,
        path=search_result.path,
        path_cost=search_result.cost,
        path_length=len(search_result.path),
        nodes_created_count=search_result.nodes_created_count,
        execution_time_ms=execution_time_ms,
    )


def run_full_benchmark(
    adjacency_matrix: pd.DataFrame,
    node_coordinates: Dict[str, Tuple[float, float]],
    slope_matrix: Optional[pd.DataFrame] = None,
    kerb_ramps_matrix: Optional[pd.DataFrame] = None,
    sidewalk_width_matrix: Optional[pd.DataFrame] = None,
    randomization_seed: Optional[int] = 42,
    test_case_count: Optional[int] = 20,
    runs_per_test_case: Optional[int] = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run comprehensive benchmarks for all available algorithms and settings.

    Args:
        adjacency_matrix (pd.DataFrame): Base adjacency matrix with distances.
        node_coordinates (Dict[str, Tuple[float, float]]): Node coordinate map.
        slope_matrix (Optional[pd.DataFrame]): Matrix of slope percentages.
        kerb_ramps_matrix (Optional[pd.DataFrame]): Matrix indicating presence of kerb ramps.
        sidewalk_width_matrix (Optional[pd.DataFrame]): Matrix of sidewalk widths.
        randomization_seed (Optional[int]): Seed for randomization. Default is 42.
        test_case_count (Optional[int]): Maximum number of test cases to run. Default is 20.
        runs_per_test_case (Optional[int]): Number of runs to average for execution time. Default is 10.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Detailed benchmark results for all test cases
            - Average results grouped by algorithm and accessibility setting
    """
    random.seed(randomization_seed)

    # Create all possible test cases and randomly sample from them
    test_cases: List[TestCase] = []
    for start, goal in itertools.combinations(adjacency_matrix.index, 2):
        test_cases.append(
            TestCase(
                adjacency_matrix=adjacency_matrix,
                node_coordinates=node_coordinates,
                slope_matrix=slope_matrix,
                kerb_ramps_matrix=kerb_ramps_matrix,
                sidewalk_width_matrix=sidewalk_width_matrix,
                start=start,
                goal=goal,
            )
        )
    if len(test_cases) > test_case_count:
        test_cases = random.sample(test_cases, test_case_count)

    results: List[BenchmarkResult] = []
    for algorithm in Algorithm:
        for test_case in test_cases:
            for consider_accessibility in (True, False):
                test_case.consider_accessibility = consider_accessibility

                # Run multiple times and average execution time
                execution_times = []
                result = None
                for _ in range(runs_per_test_case):
                    result = run_benchmark(algorithm, test_case)
                    execution_times.append(result["execution_time_ms"])

                # Update the execution time with the average
                result["execution_time_ms"] = sum(execution_times) / len(
                    execution_times
                )
                results.append(result)

    detailed_results = pd.DataFrame(results)
    average_results = (
        detailed_results.groupby(["algorithm", "consider_accessibility"])
        .agg(
            {
                "path_cost": "mean",
                "path_length": "mean",
                "nodes_created_count": "mean",
                "execution_time_ms": "mean",
            }
        )
        .reset_index()
    )

    return detailed_results, average_results
