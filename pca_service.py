import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Tuple, List
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns # <-- NEW IMPORT
import io
import base64


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect numeric columns suitable for PCA.
    Excludes ID columns and columns with too many missing values.
    """
    numeric_cols = []
    
    for col in df.columns:
        # Skip if column name suggests it's an ID or index
        col_lower = str(col).lower()
        if any(x in col_lower for x in ['id','label', 'index', 'name', 'cluster', 'group', 'pc']):
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


def run_pca(
    df: pd.DataFrame,
    oxide_cols: List[str] = None,
    normalize: bool = True,
    scale_features: bool = True,
    n_components: int = 5 # Increased default components for more analysis options
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Perform PCA on the dataframe.
    
    Args:
        df: Input dataframe
        oxide_cols: Specific columns to use. If None, auto-detect numeric columns.
        normalize: Whether to normalize rows to sum to 1 (default: True)
        scale_features: Whether to standardize features (default: True)
        n_components: Number of principal components (max 5)
    
    Returns:
        Tuple of (scores, explained_variance, loadings_df)
    """
    # Auto-detect columns if not provided
    if oxide_cols is None:
        oxide_cols = detect_numeric_columns(df)
    
    if not oxide_cols:
        raise ValueError("No numeric columns found for PCA")
    
    # Limit n_components to min(5, actual features)
    max_possible_components = min(5, len(oxide_cols))
    if n_components > max_possible_components:
        n_components = max_possible_components
    
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
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    explained_variance = pca.explained_variance_ratio_
    
    # Create loadings dataframe
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=oxide_cols
    )
    
    return scores, explained_variance, loadings


def create_pca_plot(
    scores: np.ndarray,
    explained_variance: np.ndarray,
    df: pd.DataFrame = None,
    pc_x: int = 0, # <-- NEW: Index for X-axis component (0 for PC1)
    pc_y: int = 1  # <-- NEW: Index for Y-axis component (1 for PC2)
) -> str:
    """
    Create PCA scatter plot for selected PCs and return as base64 encoded PNG.
    """
    if scores.shape[1] < max(pc_x, pc_y) + 1:
        # Handle case where selected PCs are out of bounds
        return "" 

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine axes labels
    pc_x_label = f'PC{pc_x + 1}'
    pc_y_label = f'PC{pc_y + 1}'
    
    # Check if there's a Group column for coloring
    if df is not None and "Group" in df.columns:
        groups = df["Group"].fillna("Unknown").values
        unique_groups = sorted(set(groups))
        colors_map = {g: plt.cm.tab20(i % 20) for i, g in enumerate(unique_groups)}
        
        for g in unique_groups:
            idx = (groups == g)
            ax.scatter(
                scores[idx, pc_x], scores[idx, pc_y], # Use selected PC indices
                label=str(g), s=50,
                color=colors_map[g],
                edgecolors='black',
                linewidths=0.5,
                alpha=0.7
            )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Group", fontsize=9)
    else:
        ax.scatter(
            scores[:, pc_x], scores[:, pc_y], # Use selected PC indices
            s=50,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.7,
            color='#2b88ff'
        )
    
    # Labels with explained variance
    ax.set_xlabel(f'{pc_x_label} ({explained_variance[pc_x]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'{pc_y_label} ({explained_variance[pc_y]*100:.1f}%)', fontsize=11)
    ax.set_title(f'PCA Scatter Plot: {pc_x_label} vs {pc_y_label}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig) # Use fig object for closing
    
    return img_base64

# --- NEW FUNCTION FOR HEATMAP ---
def create_loadings_heatmap(loadings_df: pd.DataFrame) -> str:
    """
    Create a heatmap of PCA loadings and return as base64 encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(max(8, loadings_df.shape[1] * 1.5), max(6, loadings_df.shape[0] * 0.5)))
    
    sns.heatmap(
        loadings_df, 
        annot=True, 
        cmap="coolwarm", 
        center=0, 
        fmt=".2f",
        cbar_kws={"label": "Loading Value"}, 
        ax=ax,
        linewidths=0.5,
        linecolor='black'
    )
    
    ax.set_title("PCA Loadings Heatmap: Oxide Contributions", fontsize=14, fontweight='bold')
    ax.set_xlabel("Principal Components", fontsize=12)
    ax.set_ylabel("Oxides", fontsize=12)
    
    # Convert plot to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    
    return img_base64


def process_pca_job(
    input_file: Path,
    output_dir: Path,
    n_components: int = 5 # Use higher default
) -> dict:
    """
    Process a PCA job from file input to file output.
    """
    # Read input file
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    elif input_file.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    # Run PCA
    scores, explained_variance, loadings = run_pca(
        df,
        n_components=n_components
    )
    
    # Update n_components in case it was capped by the number of features
    n_components = scores.shape[1] 

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scores CSV
    scores_df = df.copy()
    for i in range(n_components):
        scores_df[f'PC{i+1}'] = scores[:, i]
    scores_path = output_dir / "pca_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    
    # Save loadings CSV
    loadings_path = output_dir / "pca_loadings.csv"
    loadings.to_csv(loadings_path)
    
    # Save explained variance
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(n_components)],
        'Explained_Variance': explained_variance,
        'Cumulative_Variance': np.cumsum(explained_variance)
    })
    variance_path = output_dir / "explained_variance.csv"
    variance_df.to_csv(variance_path, index=False)
    
    # Create and save default PC1 vs PC2 plot
    plot_base64_pc12 = create_pca_plot(scores, explained_variance, df, pc_x=0, pc_y=1)
    plot_path_pc12 = output_dir / "pca_plot_pc1_vs_pc2.png"
    with open(plot_path_pc12, 'wb') as f:
        f.write(base64.b64decode(plot_base64_pc12))

    # Create and save heatmap plot
    heatmap_base64 = create_loadings_heatmap(loadings)
    heatmap_path = output_dir / "pca_heatmap_loadings.png"
    with open(heatmap_path, 'wb') as f:
        f.write(base64.b64decode(heatmap_base64))
    
    # Prepare metadata
    metadata = {
        "n_components": n_components,
        "n_samples": len(df),
        "explained_variance": explained_variance.tolist(),
        "cumulative_variance": np.cumsum(explained_variance).tolist(),
        "columns_used": loadings.index.tolist(),
        "scores_file": str(scores_path),
        "loadings_file": str(loadings_path),
        "variance_file": str(variance_path),
        "plot_file_pc12": str(plot_path_pc12),
        "heatmap_file": str(heatmap_path), # <-- NEW
        "plot_base64_pc12": plot_base64_pc12,
        "heatmap_base64": heatmap_base64     # <-- NEW
    }
    
    return metadata