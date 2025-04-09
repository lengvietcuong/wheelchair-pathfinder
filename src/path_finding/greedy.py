"""
Greedy Pathfinding Algorithm Implementation.
"""

import heapq
import logging
from typing import List

from .custom_types import Move, SearchAction, SearchResult, SearchStep
from .path_finder import PathFinder


logger = logging.getLogger(__name__)


class Greedy(PathFinder):
    """Greedy pathfinding algorithm with optional wheelchair accessibility consideration."""

    def find_path(
        self,
        start: str,
        goal: str,
        consider_accessibility: bool = True,
    ) -> SearchResult:
        if start not in self._graph or goal not in self._graph:
            raise ValueError("Start or goal location not found")

        self._reset_states()
        # Choose appropriate adjacency matrix and heuristic cost multiplier
        if consider_accessibility:
            logger.debug(
                f"Accessibility considered. Using adjusted adjacency matrix and cost multiplier ({self._accessibility_cost_multiplier}x)"
            )
            adjacency_matrix = self._accessibility_adjacency_matrix
            heuristic_cost_multiplier = self._accessibility_cost_multiplier
        else:
            logger.debug(
                "Accessibility ignored. Using base adjacency matrix and cost multiplier"
            )
            adjacency_matrix = self._base_adjacency_matrix
            heuristic_cost_multiplier = 1.0

        # Initialize frontier with the start node
        frontier: List[Move] = [Move(source=None, destination=start)]
        # Record start node being added to frontier
        self._steps.append(SearchStep(action=SearchAction.ADD_TO_FRONTIER, node=start))
        
        while frontier:
            move = heapq.heappop(frontier)
            logger.debug(f"Exploring move from '{move.source}' to '{move.destination}'")
            # Record node being explored
            self._steps.append(SearchStep(action=SearchAction.EXPLORE, node=move.destination))
            
            if move.destination in self._came_from:
                logger.debug(f"Already visited '{move.destination}', skipping")
                continue

            self._came_from[move.destination] = move.source
            if move.destination == goal:
                logger.debug("Found goal node. Returing result")
                path = self._reconstruct_path(goal)
                cost = sum(
                    adjacency_matrix.loc[path[i], path[i + 1]]
                    for i in range(len(path) - 1)
                )
                return SearchResult(
                    path=path,
                    cost=cost,
                    nodes_created_count=len(self._nodes_created),
                    steps=self._steps,
                )

            for new_move in self._expand(move.destination):
                neighbor = new_move.destination
                # Use just the heuristic cost to prioritize moves
                new_move.priority = (
                    self._get_euclidean_distance(neighbor, goal)
                    * heuristic_cost_multiplier
                )
                heapq.heappush(frontier, new_move)
                # Record node being added to frontier
                self._steps.append(SearchStep(action=SearchAction.ADD_TO_FRONTIER, node=neighbor))
                logger.debug(
                    f"Added move from '{new_move.source}' to '{new_move.destination}' to explore next. Priority: {new_move.priority}"
                )

        logger.debug(f"No path found (explored {len(self._came_from)} nodes)")
        return SearchResult(
            path=[], 
            cost=float("inf"), 
            nodes_created_count=len(self._nodes_created),
            steps=self._steps,
        )
