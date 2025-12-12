"""
Clustering service for archaeological data analysis.
Performs K-means clustering on numeric columns with automatic preprocessing.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Tuple, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect numeric columns suitable for clustering.
    Excludes ID columns and columns with too many missing values.
    """
    numeric_cols = []
    
    for col in df.columns:
        # Skip if column name suggests it's an ID or index
        col_lower = col.lower()
        if any(x in col_lower for x in ['id', 'index', 'name', 'cluster']):
            continue
        
        # Check if column is numeric or can be converted
        try:
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            # Only include if less than 50% missing values
            if numeric_data.notna().sum() / len(df) >= 0.5:
                numeric_cols.append(col)
        except Exception:
            continue
    
    return numeric_cols


def create_cluster_plot(X: np.ndarray, labels: np.ndarray, n_clusters: int, algo: str) -> str:
    """
    Create a scatter plot of clusters using PCA for 2D visualization.
    
    Args:
        X: Preprocessed feature matrix
        labels: Cluster labels
        n_clusters: Number of clusters
        algo: Algorithm name for title
    
    Returns:
        Base64 encoded PNG image string
    """
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    
    plt.figure(figsize=(10, 8))
    
# Plot each cluster with different color
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    for i in range(n_clusters):
        mask = labels == i
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=[colors[i]], label=f'Cluster {i}', 
                   alpha=0.6, edgecolors='w', s=100)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    plt.title(f'Cluster Visualization ({algo})', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def run_clustering(
    df: pd.DataFrame,
    n_clusters: int = 3,
    algo: str = "KMeans",
    normalize: bool = True,
    scale_features: bool = True,
    oxide_cols: List[str] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Perform clustering on the dataframe.
    
    Args:
        df: Input dataframe
        n_clusters: Number of clusters (default: 3)
        algo: Clustering algorithm - "KMeans" or "Agglomerative" (default: "KMeans")
        normalize: Whether to normalize rows to sum to 1 (default: True)
        scale_features: Whether to standardize features (default: True)
        oxide_cols: Specific columns to use. If None, auto-detect numeric columns.
    
    Returns:
        Tuple of (clustered dataframe, metadata dict)
    """
    # Auto-detect columns if not provided
    if oxide_cols is None:
        oxide_cols = detect_numeric_columns(df)
    
    if not oxide_cols:
        raise ValueError("No numeric columns found for clustering")
    
    # Extract and prepare data
    X = df[oxide_cols].fillna(0).values.astype(float)
    
    # Normalize rows (each row sums to 1)
    if normalize:
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        X = X / row_sums
    
    # Standardize features (mean=0, std=1)
    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Perform clustering based on algorithm choice
    if algo == "KMeans":
        from sklearn.cluster import KMeans
        labels = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit_predict(X)
    elif algo == "Agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Use 'KMeans' or 'Agglomerative'")
    
    # Add cluster labels to dataframe
    result_df = df.copy()
    result_df["Cluster"] = labels
    
    # Generate visualization plot
    plot_base64 = create_cluster_plot(X, labels, n_clusters, algo)
    
    # Prepare metadata
    metadata = {
        "n_clusters": n_clusters,
        "columns_used": oxide_cols,
        "n_samples": len(df),
        "normalize": normalize,
        "scale_features": scale_features,
        "algorithm": algo,
        "plot_base64": plot_base64
    }
    
    return result_df, metadata


def process_clustering_job(
    input_file: Path,
    output_file: Path,
    n_clusters: int = 3,
    algo: str = "KMeans"
) -> dict:
    """
    Process a clustering job from file input to file output.
    
    Args:
        input_file: Path to input CSV or Excel file
        output_file: Path to save clustered CSV
        n_clusters: Number of clusters
        algo: Clustering algorithm - "KMeans" or "Agglomerative"
    
    Returns:
        Metadata dictionary with processing information
    """
    # Read input file
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    elif input_file.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    # Run clustering
    clustered_df, metadata = run_clustering(df, n_clusters=n_clusters, algo=algo)
    
    # Save output CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    clustered_df.to_csv(output_file, index=False)
    
    # Save plot PNG
    plot_path = output_file.parent / "cluster_plot.png"
    plot_base64 = metadata["plot_base64"]
    plot_bytes = base64.b64decode(plot_base64)
    with open(plot_path, "wb") as f:
        f.write(plot_bytes)
    
    metadata["input_file"] = str(input_file)
    metadata["output_file"] = str(output_file)
    metadata["plot_file"] = str(plot_path)
    
    return metadata
