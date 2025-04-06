"""
KMZ to Adjacency Matrix Converter.
"""

import logging
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from geopy.distance import geodesic


TEMP_DIR = Path("temp_kmz")
KML_BASENAME = "doc.kml"
DEFAULT_KMZ_PATH = Path("data") / "map.kmz"
OUTPUT_CSV = Path("data") / "adjacency_matrix.csv"
KML_NAMESPACE = {"kml": "http://www.opengis.net/kml/2.2"}
MIN_PATH_SEGMENTS = 20
# Maximum distance in meters to consider a path passing through a node
NODE_PROXIMITY_THRESHOLD = 50


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_kmz_to_kml(kmz_path: Path, output_dir: Path = TEMP_DIR) -> Path:
    """
    Extract a KMZ archive and return the path to its KML file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(kmz_path, "r") as zip_reference:
        zip_reference.extractall(output_dir)
    kml_path = output_dir / KML_BASENAME
    return kml_path


def parse_kml_file(
    kml_path: Path,
) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Parse the KML file, extracting point‐placemarks as nodes and line‐placemarks as paths.

    Args:
        kml_path (Path): Path to the KML file.

    Returns:
        Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing:
            - A dictionary of nodes with IDs as keys and names and coordinates as values.
            - A list of paths with names and coordinates.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    nodes_dict: Dict[int, Dict[str, Any]] = {}
    paths_list: List[Dict[str, Any]] = []
    next_node_id = 0

    placemarks = root.findall(".//kml:Placemark", KML_NAMESPACE)
    for placemark in placemarks:
        name_element = placemark.find("kml:name", KML_NAMESPACE)
        place_name = (
            name_element.text if name_element is not None else f"Node_{next_node_id}"
        )

        point_element = placemark.find(".//kml:Point", KML_NAMESPACE)
        line_element = placemark.find(".//kml:LineString", KML_NAMESPACE)

        if point_element is not None:
            coords_element = point_element.find("kml:coordinates", KML_NAMESPACE)
            if coords_element is None:
                continue

            lon, lat, _ = map(float, coords_element.text.strip().split(","))
            nodes_dict[next_node_id] = {"name": place_name, "coords": (lat, lon)}
            next_node_id += 1

        elif line_element is not None:
            coords_element = line_element.find("kml:coordinates", KML_NAMESPACE)
            if coords_element is None:
                continue

            coordinate_pairs = []
            for raw_coord in coords_element.text.strip().split():
                lon, lat, _ = map(float, raw_coord.split(","))
                coordinate_pairs.append((lat, lon))

            paths_list.append({"name": place_name, "coordinates": coordinate_pairs})

    return nodes_dict, paths_list


def find_nodes_on_path(
    path_coords: List[Tuple[float, float]],
    nodes: Dict[int, Dict[str, Any]],
    threshold: float = NODE_PROXIMITY_THRESHOLD,
) -> List[Tuple[int, int, float]]:
    """
    Find nodes that lie on a path and return the connections between them.

    Args:
        path_coords: List of coordinates forming the path
        nodes: Dictionary of nodes with coordinates
        threshold: Maximum distance in meters to consider a node as being on the path

    Returns:
        List of tuples (start_node_id, end_node_id, distance) representing path segments
    """
    # Find all points where the path intersects with nodes
    intersections = []
    for i, coord in enumerate(path_coords):
        for node_id, node_data in nodes.items():
            distance = geodesic(coord, node_data["coords"]).meters
            if distance <= threshold:
                intersections.append((i, node_id, coord))

    # Sort intersections by position along the path
    intersections.sort(key=lambda x: x[0])

    # If fewer than 2 intersections, no valid path segments
    if len(intersections) < 2:
        return []

    # Create path segments between consecutive intersected nodes
    path_segments = []
    for i in range(len(intersections) - 1):
        start_idx, start_node, _ = intersections[i]
        end_idx, end_node, _ = intersections[i + 1]

        # Skip if it's the same node
        if start_node == end_node:
            continue

        # Calculate actual path distance between the nodes
        segment_distance = 0.0
        for j in range(start_idx, end_idx):
            segment_distance += geodesic(path_coords[j], path_coords[j + 1]).meters

        path_segments.append((start_node, end_node, segment_distance))

    return path_segments


def create_adjacency_matrix(
    nodes: Dict[int, Dict[str, Any]], paths: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Build a symmetric adjacency‐distance matrix (in meters) for all nodes.
    Uses improved algorithm to only connect nodes that are actually connected by paths.
    """
    node_count = len(nodes)
    matrix = np.full((node_count, node_count), np.inf)
    np.fill_diagonal(matrix, 0.0)

    for path in paths:
        coords = path["coordinates"]

        # Find connections between nodes that actually exist on this path
        path_segments = find_nodes_on_path(coords, nodes)

        for start_id, end_id, distance in path_segments:
            # Update the matrix if this path provides a shorter connection
            if distance < matrix[start_id, end_id]:
                matrix[start_id, end_id] = distance
                matrix[end_id, start_id] = distance  # Ensure symmetry

    node_labels = [nodes[i]["name"] for i in range(node_count)]
    adjacency_df = pd.DataFrame(matrix, index=node_labels, columns=node_labels)
    return adjacency_df


def main():
    kmz_path = DEFAULT_KMZ_PATH

    try:
        kml_path = extract_kmz_to_kml(kmz_path)
        logger.info("KMZ extracted to KML successfully.")

        nodes, paths = parse_kml_file(kml_path)
        logger.info(f"Parsed {len(nodes)} nodes and {len(paths)} paths from KML.")

        if len(paths) < MIN_PATH_SEGMENTS:
            logger.warning(
                f"Only {len(paths)} path segments found; "
                f"expected at least {MIN_PATH_SEGMENTS}."
            )

        adjacency_df = create_adjacency_matrix(nodes, paths)
        adjacency_df.to_csv(OUTPUT_CSV)
        logger.info(f"Adjacency matrix saved to '{OUTPUT_CSV}'.")

    except Exception as error:
        logger.error(f"Error processing KMZ file: {error}")

    finally:
        # Remove the entire temp directory
        if TEMP_DIR.exists() and TEMP_DIR.is_dir():
            shutil.rmtree(TEMP_DIR)
            logger.info(f"Temporary directory '{TEMP_DIR}' removed.")


if __name__ == "__main__":
    main()
