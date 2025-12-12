import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import base64
from io import BytesIO
from collections import defaultdict

# Ternary diagram map: system name -> (image path relative to BASE_DIR, extent)
TERNARY_MAP = {
    "CaO–SiO2–Al2O3": ("ternary/CaAlSi Ox clean str.jpg", [-5.5, 105, -10.5, 95]),
    "K2O–SiO2–Al2O3": ("ternary/KAlSiOx clean.jpg", [-2, 103, -5, 87]),
    "FeO–SiO2–Al2O3": ("ternary/FeO SiO2 Al2O3 clean.jpg", [-3, 106, -23.5, 90]),
    "MnO–SiO2–Al2O3": ("ternary/Al2O3 MnO SiO2 clean.jpg", [-21.67, 112.6, -13.5, 97.3]),
    "FeO–CaO–SiO2": ("ternary/CaO-FeO-SiO2 clean.jpg", [-6, 104, -7.8, 96]),
}


def v(row, key):
    """Safely retrieve and convert oxide wt%."""
    try:
        # Convert key to uppercase for consistent lookup
        return float(row.get(key.upper(), 0.0))
    except (ValueError, TypeError):
        return 0.0


def choose_ternary(row, include_k2o=True, ternary_map=None):
    """
    Rule-based ternary system selection based on oxide dominance.
    """
    if ternary_map is None:
        return None

    CaO, FeO, MnO, MgO, Al2O3, SiO2, TiO2, K2O_val = (
        v(row, "CaO"), v(row, "FeO"), v(row, "MnO"), v(row, "MgO"),
        v(row, "Al2O3"), v(row, "SiO2"), v(row, "TIO2"), (v(row, "K2O") if include_k2o else 0.0)
    )

    C_SiO2_Group = SiO2 + TiO2
    C_Al2O3_Group = Al2O3
    minor_basics = MnO + MgO + K2O_val

    system = None
    C1, C2, C3 = 0, 0, 0
    corner_names = ("", "", "")
    
    if Al2O3 <= 5 and CaO > 1 and FeO > 1:
        system = "FeO–CaO–SiO2"
        C_FeO_Group = FeO + minor_basics + C_Al2O3_Group
        C_CaO_Group = CaO
        C1 = C_SiO2_Group
        C2 = C_FeO_Group
        C3 = C_CaO_Group
        corner_names = ("FeO Group", "SiO₂ Group", "CaO")
          
    elif include_k2o and (K2O_val > CaO or K2O_val > FeO):
        system = "K2O–SiO2–Al2O3"
        C_K2O_Group = K2O_val + CaO + FeO + MnO + MgO
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, C_K2O_Group
        corner_names = ("Al₂O₃", "SiO₂ Group", "K₂O Group")

    elif CaO > FeO and CaO > MnO and CaO > MgO and CaO > K2O_val:
        system = "CaO–SiO2–Al2O3"
        C_CaO_Group = CaO + FeO + minor_basics
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, C_CaO_Group
        corner_names = ("Al₂O₃", "SiO₂ Group", "CaO Group")

    elif FeO > CaO and FeO > MnO and FeO > MgO and FeO > K2O_val:
        system = "FeO–SiO2–Al2O3"
        C_FeO_Group = FeO + CaO + minor_basics
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, C_FeO_Group
        corner_names = ("Al₂O₃", "SiO₂ Group", "FeO Group")

    elif MnO >= FeO:
        system = "MnO–SiO2–Al2O3"
        C_MnO_Group = MnO + FeO + CaO + MgO + K2O_val
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, C_MnO_Group
        corner_names = ("Al₂O₃", "SiO₂ Group", "MnO Group")

    else:
        return None

    total = C1 + C2 + C3
    if total <= 0:
        return None

    return {
        # FIX 2: Changed to uppercase "LABEL" to match DataFrame column names
        "Label": row.get("LABEL", "Sample"),
        "System": system,
        "Norm_Right": 100 * C2 / total,
        "Norm_Top": 100 * C1 / total,
        "Norm_Left": 100 * C3 / total,
        "Corner_Names": corner_names,
        "Image": ternary_map[system][0],
        "Extent": ternary_map[system][1]
    }


def manual_system(row, system, include_k2o=True):
    """
    Calculate ternary coordinates for a manually selected system.
    """
    CaO, FeO, MnO, MgO, Al2O3, SiO2, TiO2, K2O_val = (
        v(row, "CaO"), v(row, "FeO"), v(row, "MnO"), v(row, "MgO"),
        v(row, "Al2O3"), v(row, "SiO2"), v(row, "TIO2"), (v(row, "K2O") if include_k2o else 0.0)
    )
    C_SiO2_Group = SiO2 + TiO2
    C_Al2O3_Group = Al2O3
    minor_basics = MnO + MgO + K2O_val

    if system == "MnO–SiO2–Al2O3":
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, MnO + FeO + CaO + MgO + K2O_val
        corner_names = ("Al₂O₃", "SiO₂ Group", "MnO Group")
    elif system == "CaO–SiO2–Al2O3":
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, CaO + FeO + minor_basics
        corner_names = ("Al₂O₃", "SiO₂ Group", "CaO Group")
    elif system == "FeO–SiO2–Al2O3":
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, FeO + CaO + minor_basics
        corner_names = ("Al₂O₃", "SiO₂ Group", "FeO Group")
    elif system == "K2O–SiO2–Al2O3":
        C1, C2, C3 = C_SiO2_Group, C_Al2O3_Group, K2O_val + CaO + FeO + MnO + MgO
        corner_names = ("Al₂O₃", "SiO₂ Group", "K₂O Group")
    elif system == "FeO–CaO–SiO2":
        C1, C2, C3 = C_SiO2_Group, FeO + minor_basics + C_Al2O3_Group, CaO
        corner_names = ("FeO Group", "SiO₂ Group", "CaO")
    else:
        return None

    total = C1 + C2 + C3
    if total <= 0:
        return None

    return {
        # FIX 2: Changed to uppercase "LABEL" to match DataFrame column names
        "Label": row.get("LABEL", "Sample"),
        "System": system,
        "Norm_Right": 100 * C2 / total,
        "Norm_Top": 100 * C1 / total,
        "Norm_Left": 100 * C3 / total,
        "Corner_Names": corner_names,
        "Image": TERNARY_MAP[system][0],
        "Extent": TERNARY_MAP[system][1]
    }


def plot_ternary(entries, base_dir_root: str):
    """
    Create ternary plot with background image overlay using pure matplotlib.
    """
    if not entries:
        raise ValueError("No valid entries to plot")

    system = entries[0]["System"]
    
    # --- FIX: Construct ABSOLUTE path using base_dir_root ---
    image_path = Path(base_dir_root) / entries[0]["Image"]
    
    extent = entries[0]["Extent"]
    corner_names = entries[0]["Corner_Names"]

    # Extract coordinates (Right, Top, Left)
    coords = [(e["Norm_Right"], e["Norm_Top"], e["Norm_Left"]) for e in entries]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Try to load and display background image
    try:
        if image_path.exists():
            img = Image.open(image_path).convert("RGBA")
            ax.imshow(np.array(img), extent=extent, aspect='auto', zorder=-1)
        else:
            print(f"Warning: Background image not found at absolute path: {image_path}")
    except Exception as e:
        print(f"Warning: Could not load background image at {image_path}: {e}")
    
    # Convert ternary coordinates to cartesian for plotting
    x_coords = []
    y_coords = []
    for right, top, left in coords:
        # Transformation: x = r + t/2, y = t * sqrt(3)/2
        x = right + top / 2
        y = top * np.sqrt(3) / 2
        x_coords.append(x)
        y_coords.append(y)
    
    # Plot scatter points
    ax.scatter(x_coords, y_coords, color="red", marker="o", s=20, 
                edgecolor='black', zorder=5)
    
    # Turn off axis
    ax.axis('off')
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    plt.tight_layout()

    # Convert to base64
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return img_base64


def process_plotting_job(
    input_file: Path,
    output_dir: Path,
    mode: str = "auto",
    system: Optional[str] = None,
    include_k2o: bool = True,
    base_dir_root: str = "."
) -> dict:
    """
    Process ternary plotting job from file input.
    """
    # Read input file
    if input_file.suffix.lower() == '.csv':
        df = pd.read_csv(input_file)
    elif input_file.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file.suffix}")
    
    # Normalize column names to uppercase and convert to numeric
    # This is why we must search for "LABEL" and not "Label" later.
    df.columns = [col.upper() for col in df.columns]
    
    # List of oxides that need to be numeric (using uppercase names)
    slag_oxid = ['NA2O', 'MGO', 'AL2O3', 'SIO2', 'P2O5', 'SO3', 'K2O', 'CAO',
                  'TIO2', 'V2O5', 'CR2O3', 'MNO', 'FEO']
    
    for oxd in slag_oxid:
        if oxd in df.columns:
            df[oxd] = pd.to_numeric(df[oxd], errors='coerce')
    
    df = df.fillna(0.0)
    
    # Add LABEL column if not present (using uppercase name)
    if 'LABEL' not in df.columns:
        df['LABEL'] = [f"Sample_{i+1}" for i in range(len(df))]
    
    results = []
    
    # ... (rest of mode logic: auto/manual) ...
    if mode == "auto":
        # Auto mode: determine best system from dataset averages
        avg_vals = df.mean(numeric_only=True).to_dict()
        avg_vals['LABEL'] = 'Average' # Use uppercase here too for consistency
        
        best_system_entry = choose_ternary(avg_vals, include_k2o=include_k2o, ternary_map=TERNARY_MAP)
        if not best_system_entry:
            raise ValueError("Could not determine ternary system from data")
        
        selected_system = best_system_entry["System"]
        
        # Apply selected system to all rows
        for _, row in df.iterrows():
            # Use dictionary representation of row (keys are uppercase)
            res = manual_system(row.to_dict(), selected_system, include_k2o=include_k2o)
            if res:
                results.append(res)
    else:
        # Manual mode: use specified system
        if not system or system not in TERNARY_MAP:
            raise ValueError(f"Invalid system: {system}. Must be one of {list(TERNARY_MAP.keys())}")
        
        selected_system = system
        for _, row in df.iterrows():
            res = manual_system(row.to_dict(), selected_system, include_k2o=include_k2o)
            if res:
                results.append(res)
    
    if not results:
        raise ValueError("No valid data points for ternary plotting")
    
    # Create plot - PASS ABSOLUTE PATH
    plot_base64 = plot_ternary(results, base_dir_root)
    
    # Save plot to file
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "ternary_plot.png"
    plot_bytes = base64.b64decode(plot_base64)
    with open(plot_path, "wb") as f:
        f.write(plot_bytes)
    
    # Save normalized data to CSV
    normalized_csv = output_dir / "normalized_data.csv"
    normalized_df = pd.DataFrame([{
        "Label": r["Label"],
        f"{r['Corner_Names'][0]} (Right)": f"{r['Norm_Right']:.2f}",
        f"{r['Corner_Names'][1]} (Top)": f"{r['Norm_Top']:.2f}",
        f"{r['Corner_Names'][2]} (Left)": f"{r['Norm_Left']:.2f}"
    } for r in results])
    normalized_df.to_csv(normalized_csv, index=False)
    
    # Prepare simplified data for frontend display
    normalized_data_for_display = [{
        "Label": r["Label"],
        r['Corner_Names'][0]: f"{r['Norm_Right']:.2f}",
        r['Corner_Names'][1]: f"{r['Norm_Top']:.2f}",
        r['Corner_Names'][2]: f"{r['Norm_Left']:.2f}"
    } for r in results]
    
    return {
        "system": selected_system,
        "n_samples": len(results),
        "mode": mode,
        "include_k2o": include_k2o,
        "plot_base64": plot_base64,
        "plot_file": str(plot_path),
        "normalized_file": str(normalized_csv),
        "corner_names": results[0]["Corner_Names"],
        "normalized_data": normalized_data_for_display
    }