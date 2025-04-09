import concurrent.futures
import itertools
import logging
import random
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .a_star import AStar
from .custom_types import (
    BenchmarkResult,
    TestCase,
)


logger = logging.getLogger(__name__)


def run_benchmark(test_case: TestCase) -> BenchmarkResult:
    """
    Run a benchmark for the given pathfinder and test case.

    Args:
        test_case (TestCase): Test case configurations.

    Returns:
        BenchmarkResult: The result of running the benchmark.
    """
    path_finder = AStar(
        adjacency_matrix=test_case.adjacency_matrix,
        node_coordinates=test_case.node_coordinates,
        slope_matrix=test_case.slope_matrix,
        kerb_ramps_matrix=test_case.kerb_ramps_matrix,
        sidewalk_width_matrix=test_case.sidewalk_width_matrix,
    )

    start_time = time.perf_counter()
    search_result = path_finder.find_path(
        test_case.start, test_case.goal, test_case.consider_accessibility
    )
    execution_time_ms = (time.perf_counter() - start_time) * 1_000

    return BenchmarkResult(
        algorithm="A*",
        consider_accessibility=test_case.consider_accessibility,
        start=test_case.start,
        goal=test_case.goal,
        path=search_result.path,
        distance=search_result.distance,
        accessibility_cost=search_result.accessibility_cost,
        nodes_created_count=search_result.nodes_created_count,
        nodes_explored_count=search_result.nodes_explored_count,
        execution_time_ms=execution_time_ms,
    )


def run_benchmark_multiple_times(test_case: TestCase, count: int) -> BenchmarkResult:
    """
    Run a benchmark multiple times and return the averaged result.

    Args:
        test_case (TestCase): Test case configurations.
        count (int): Number of runs to average execution time over.

    Returns:
        BenchmarkResult: The result with averaged execution time.
    """
    execution_times = []
    result = None
    for _ in range(count):
        result = run_benchmark(test_case)
        execution_times.append(result["execution_time_ms"])

    # Update the execution time with the average
    result["execution_time_ms"] = sum(execution_times) / len(execution_times)
    return result


def run_full_benchmark(
    adjacency_matrix: pd.DataFrame,
    node_coordinates: Dict[str, Tuple[float, float]],
    slope_matrix: Optional[pd.DataFrame] = None,
    kerb_ramps_matrix: Optional[pd.DataFrame] = None,
    sidewalk_width_matrix: Optional[pd.DataFrame] = None,
    randomization_seed: Optional[int] = 42,
    test_case_count: Optional[int] = 20,
    runs_per_test_case: Optional[int] = 5,
    max_workers: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run comprehensive benchmarks for all available algorithms and settings in parallel.

    Args:
        adjacency_matrix (pd.DataFrame): Base adjacency matrix with distances.
        node_coordinates (Dict[str, Tuple[float, float]]): Node coordinate map.
        slope_matrix (Optional[pd.DataFrame]): Matrix of slope percentages.
        kerb_ramps_matrix (Optional[pd.DataFrame]): Matrix indicating presence of kerb ramps.
        sidewalk_width_matrix (Optional[pd.DataFrame]): Matrix of sidewalk widths.
        randomization_seed (Optional[int]): Seed for randomization. Default is 42.
        test_case_count (Optional[int]): Maximum number of test cases to run. Default is 20.
        runs_per_test_case (Optional[int]): Number of runs to average for execution time. Default is 5.
        max_workers (Optional[int]): Maximum number of worker processes to use. None means use all available.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Detailed benchmark results for all test cases
            - Average results grouped by algorithm and accessibility setting
    """
    random.seed(randomization_seed)

    logger.debug("Generating test cases...")
    # Create (start, goal) pairs
    locations = list(itertools.combinations(adjacency_matrix.index, 2))
    if len(locations) > test_case_count:
        locations = random.sample(locations, test_case_count)
    # Create test cases
    test_cases: List[TestCase] = []
    for start, goal in locations:
        for consider_accessibility in (True, False):
            test_cases.append(
                TestCase(
                    adjacency_matrix=adjacency_matrix,
                    node_coordinates=node_coordinates,
                    slope_matrix=slope_matrix,
                    kerb_ramps_matrix=kerb_ramps_matrix,
                    sidewalk_width_matrix=sidewalk_width_matrix,
                    start=start,
                    goal=goal,
                    consider_accessibility=consider_accessibility,
                )
            )

    logger.debug("Running benchmarks...")
    # Run benchmarks in parallel
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                run_benchmark_multiple_times,
                test_case,
                runs_per_test_case,
            ): test_case
            for test_case in test_cases
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
            except Exception as error:
                task = future_to_task[future]
                logger.error(f"Error with task {task}: {error}", exc_info=True)

    logger.debug("Benchmarking completed. Compiling results...")
    detailed_results = pd.DataFrame(results)
    # When calculating the average, ignore unreachable paths
    valid_paths = detailed_results[detailed_results["distance"] != float("inf")]
    average_results = (
        valid_paths.groupby(["algorithm", "consider_accessibility"])
        .agg(
            {
                "distance": "mean",
                "accessibility_cost": "mean",
                "nodes_created_count": "mean",
                "nodes_explored_count": "mean",
                "execution_time_ms": "mean",
            }
        )
        .round(2)
        .reset_index()
        .rename(
            columns={
                "algorithm": "Alg",
                "consider_accessibility": "Accessibility",
                "distance": "Dist (m)",
                "accessibility_cost": "Wheelchair cost",
                "nodes_created_count": "Nodes created",
                "nodes_explored_count": "Nodes explored",
                "execution_time_ms": "Time (ms)",
            }
        )
    )

    return detailed_results, average_results
