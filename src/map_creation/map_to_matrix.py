"""
.kmz File to Adjacency Matrix Converter.
"""

import json
import logging
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple, TypedDict

import numpy as np
import pandas as pd
from geopy.distance import geodesic


class Coordinates(NamedTuple):
    """Represents a geographical coordinate with latitude and longitude."""

    latitude: float
    longitude: float


class Node(TypedDict):
    """Represents a point-placemark node with name and coordinates."""

    name: str
    coordinates: Coordinates


class Edge(TypedDict):
    """Represents a direct connection between two nodes."""

    start_node_id: int
    end_node_id: int
    distance: float


TEMP_DIRECTORY = Path("temp_kmz")
DATA_DIRECTORY = Path("data")

KMZ_MAP_PATH = DATA_DIRECTORY / "map.kmz"
ADJACENCY_MATRIX_PATH = DATA_DIRECTORY / "adjacency_matrix.csv"
NODES_COORDINATES_PATH = DATA_DIRECTORY / "node_coordinates.json"

KML_BASENAME = "doc.kml"
KML_NAMESPACE = {"kml": "http://www.opengis.net/kml/2.2"}


logger = logging.getLogger(__name__)


def extract_kmz_to_kml(kmz_path: Path, output_directory: Path = TEMP_DIRECTORY) -> Path:
    """
    Extract a KMZ archive and return the path to its KML file.

    Args:
        kmz_path (Path): Path to the KMZ file.
        output_directory (Path): Directory to extract the KML file to.

    Returns:
        Path: Path to the extracted KML file.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(kmz_path, "r") as zip_reference:
        zip_reference.extractall(output_directory)
    kml_path = output_directory / KML_BASENAME
    return kml_path


def parse_kml_file(kml_path: Path) -> Tuple[Dict[int, Node], List[Edge]]:
    """
    Parse the KML file, extracting point-placemarks as nodes and line-placemarks as edges.
    Each line is treated as connecting exactly 2 nodes.

    Args:
        kml_path (Path): Path to the KML file.

    Returns:
        Tuple[Dict[int, Node], List[Edge]]: A tuple containing:
            - A dictionary of nodes with IDs as keys.
            - A list of edges connecting pairs of nodes.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()

    nodes_dict: Dict[int, Node] = {}
    edges_list: List[Edge] = []
    node_id = 0

    # First pass: collect all nodes
    placemarks = root.findall(".//kml:Placemark", KML_NAMESPACE)
    for placemark in placemarks:
        name_element = placemark.find("kml:name", KML_NAMESPACE)
        name = name_element.text if name_element is not None else f"Node_{node_id}"

        point_element = placemark.find(".//kml:Point", KML_NAMESPACE)
        if point_element is not None:
            coordinates_element = point_element.find("kml:coordinates", KML_NAMESPACE)
            if coordinates_element is None:
                continue

            longitude, latitude, _ = map(
                float, coordinates_element.text.strip().split(",")
            )
            nodes_dict[node_id] = {
                "name": name,
                "coordinates": (latitude, longitude),
            }
            node_id += 1

    # Second pass: collect all edges
    for placemark in placemarks:
        name_element = placemark.find("kml:name", KML_NAMESPACE)
        name = name_element.text if name_element is not None else "Edge"

        line_element = placemark.find(".//kml:LineString", KML_NAMESPACE)
        if line_element is not None:
            coordinates_element = line_element.find("kml:coordinates", KML_NAMESPACE)
            if coordinates_element is None:
                continue

            # Parse coordinates
            coordinates = []
            for raw_coordinate in coordinates_element.text.strip().split():
                longitude, latitude, _ = map(float, raw_coordinate.split(","))
                coordinates.append((latitude, longitude))

            if len(coordinates) < 2:
                logger.warning(f"Path {name} has fewer than 2 points, skipping.")
                continue

            start_coordinates = coordinates[0]
            end_coordinates = coordinates[-1]

            # Find closest nodes to start and end points
            start_node_id = find_closest_node(start_coordinates, nodes_dict)
            end_node_id = find_closest_node(end_coordinates, nodes_dict)

            if start_node_id == end_node_id:
                logger.warning(f"Path {name} connects a node to itself, skipping.")
                continue

            # Calculate direct distance
            distance = geodesic(
                nodes_dict[start_node_id]["coordinates"],
                nodes_dict[end_node_id]["coordinates"],
            ).meters

            edges_list.append(
                {
                    "start_node_id": start_node_id,
                    "end_node_id": end_node_id,
                    "distance": distance,
                }
            )

    return nodes_dict, edges_list


def find_closest_node(coordinates: Coordinates, nodes: Dict[int, Node]) -> int:
    """
    Find the node closest to the given coordinates.

    Args:
        coordinates: Latitude and longitude to check
        nodes: Dictionary of nodes with their data

    Returns:
        The ID of the closest node
    """
    closest_node_id = -1
    min_distance = float("inf")

    for node_id, node_data in nodes.items():
        distance = geodesic(coordinates, node_data["coordinates"]).meters
        if distance < min_distance:
            min_distance = distance
            closest_node_id = node_id

    return closest_node_id


def create_adjacency_matrix(nodes: Dict[int, Node], edges: List[Edge]) -> pd.DataFrame:
    """
    Build a symmetric adjacency-distance matrix (in meters) for all nodes.

    Args:
        nodes: Dictionary of nodes with their data
        edges: List of edges connecting pairs of nodes

    Returns:
        DataFrame containing the adjacency matrix with node names as indices
    """
    node_count = len(nodes)
    matrix = np.full((node_count, node_count), np.inf)
    np.fill_diagonal(matrix, 0.0)

    for edge in edges:
        start_node_id = edge["start_node_id"]
        end_node_id = edge["end_node_id"]
        distance = edge["distance"]

        # Update the matrix if this edge provides a shorter connection
        if distance < matrix[start_node_id, end_node_id]:
            matrix[start_node_id, end_node_id] = distance
            matrix[end_node_id, start_node_id] = distance  # Ensure symmetry

    node_labels = [nodes[i]["name"] for i in range(node_count)]
    adjacency_matrix_df = pd.DataFrame(matrix, index=node_labels, columns=node_labels)
    return adjacency_matrix_df


def extract_kmz_file():
    """
    Parse a KMZ file and save the adjacency matrix and node coordinates.
    """
    try:
        if not KMZ_MAP_PATH.exists():
            raise FileNotFoundError(f"KMZ file not found at '{KMZ_MAP_PATH}'.")

        # Extract map data from KMZ file
        kml_path = extract_kmz_to_kml(KMZ_MAP_PATH)
        nodes, edges = parse_kml_file(kml_path)

        # Create and save the adjacency matrix
        adjacency_matrix_df = create_adjacency_matrix(nodes, edges)
        adjacency_matrix_df.to_csv(ADJACENCY_MATRIX_PATH)
        logger.info(f"Adjacency matrix saved to '{ADJACENCY_MATRIX_PATH}'.")

        # Save node coordinates
        node_coordinates = {nodes[i]["name"]: nodes[i]["coordinates"] for i in nodes}
        with open(NODES_COORDINATES_PATH, "w") as file:
            json.dump(node_coordinates, file, indent=4)
        logger.info(f"Node coordinates saved to '{NODES_COORDINATES_PATH}'.")
    finally:
        # Clean up the temporary directory
        if TEMP_DIRECTORY.exists() and TEMP_DIRECTORY.is_dir():
            shutil.rmtree(TEMP_DIRECTORY)
