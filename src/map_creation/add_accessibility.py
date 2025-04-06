import logging
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def enhance_map_with_accessibility(
    adjacency_matrix_file: str, output_prefix: str, seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Enhance a map's adjacency matrix with random accessibility features and save to CSV files.

    Parameters:
        adjacency_matrix_file (str): Path to CSV file with adjacency matrix.
        output_prefix (str): Prefix for output CSV filenames.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Original and modified matrices plus node features.
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load adjacency matrix and replace 'inf'
    df = pd.read_csv(adjacency_matrix_file, index_col=0).replace("inf", np.inf)
    locations = df.index.tolist()

    # Initialize feature matrices
    slope_df = pd.DataFrame(index=locations, columns=locations)
    kerb_ramps_df = pd.DataFrame(index=locations, columns=locations)
    sidewalk_width_df = pd.DataFrame(index=locations, columns=locations)

    # Populate feature matrices
    for i in locations:
        for j in locations:
            dist = df.loc[i, j]
            if dist == np.inf or dist <= 0:
                continue
            slope_df.loc[i, j] = round(random.uniform(0, 15), 1)
            kerb_ramps_df.loc[i, j] = random.randint(0, 1)
            sidewalk_width_df.loc[i, j] = round(random.uniform(0.9, 2.5), 1)

    # Generate node-level accessibility features
    node_features = {
        loc: {
            "has_accessible_restroom": random.choice([True, False]),
            "has_accessible_parking": random.choice([True, False]),
            "has_accessible_entrance": random.choice([True, False]),
            "has_rest_area": random.choice([True, False]),
        }
        for loc in locations
    }
    node_features_df = pd.DataFrame.from_dict(node_features, orient="index")

    # Compile results
    results: Dict[str, pd.DataFrame] = {
        "slope": slope_df,
        "kerb_ramps": kerb_ramps_df,
        "sidewalk_width": sidewalk_width_df,
        "node_features": node_features_df,
    }

    # Save each DataFrame to CSV
    output_dir = Path(output_prefix).parent
    prefix = Path(output_prefix).stem
    for name, table in results.items():
        filename = output_dir / f"{prefix}_{name}.csv"
        table.to_csv(filename)
        logging.info("Saved %s", filename)
    return results


def main(seed: Optional[int] = 42) -> None:
    input_file = Path("data") / "adjacency_matrix.csv"
    output_prefix = Path("data") / "adjacency_matrix"
    enhance_map_with_accessibility(input_file, output_prefix, seed)


if __name__ == "__main__":
    main()
