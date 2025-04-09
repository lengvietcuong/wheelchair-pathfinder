"""
Dijkstra Pathfinding Algorithm Implementation.
"""

import heapq
import logging
from typing import Dict, List

from .custom_types import Move, SearchAction, SearchResult, SearchStep
from .path_finder import PathFinder


logger = logging.getLogger(__name__)


class Dijkstra(PathFinder):
    """Dijkstra pathfinding algorithm with optional wheelchair accessibility consideration."""

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
        else:
            logger.debug(
                "Accessibility ignored. Using base adjacency matrix and cost multiplier"
            )
            adjacency_matrix = self._base_adjacency_matrix

        # Track known costs
        g_scores: Dict[str, float] = {node: float("inf") for node in self._graph}
        g_scores[start] = 0.0
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
                return SearchResult(
                    path=self._reconstruct_path(goal),
                    cost=g_scores[goal],
                    nodes_created_count=len(self._nodes_created),
                    steps=self._steps,
                )

            for new_move in self._expand(move.destination):
                neighbor = new_move.destination
                g_score = (
                    g_scores[move.destination]
                    + adjacency_matrix.loc[move.destination, neighbor]
                )
                if g_score >= g_scores[neighbor]:
                    continue

                g_scores[neighbor] = g_score
                new_move.priority = g_score
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
