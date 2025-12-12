from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sqlite3, os, uuid, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import base64

# Ensure the current directory is in Python path for imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR) 

from pdf_service import extract_pdf_with_ocr
from clustering_service import process_clustering_job

from pca_service import process_pca_job, create_pca_plot
from plotting_service import process_plotting_job, TERNARY_MAP
from img_analysis import analyze_image 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "uploads.db")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="ArQave")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=os.path.join(BASE_DIR, "assets")), name="assets")
app.mount("/pages-css", StaticFiles(directory=os.path.join(BASE_DIR, "pages-css")), name="pages-css")
app.mount("/pages-js", StaticFiles(directory=os.path.join(BASE_DIR, "pages-js")), name="pages-js")
app.mount("/ternary", StaticFiles(directory=os.path.join(BASE_DIR, "ternary")), name="ternary")


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS uploads(
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            excel_path TEXT,
            status TEXT DEFAULT 'pending',
            created_at TEXT,
            error_msg TEXT,
            extraction_details TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


@app.get("/")
def root():
    return RedirectResponse(url="/index.html")


@app.get("/health")
def health():
    try:
        conn = db()
        conn.execute("SELECT 1")
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def update_status(job_id: str, status: str, excel_path: str | None = None, error: str | None = None, details: str | None = None):
    conn = db()
    c = conn.cursor()
    c.execute("UPDATE uploads SET status=?, excel_path=?, error_msg=?, extraction_details=? WHERE id=?",
              (status, excel_path, error, details, job_id))
    conn.commit()
    conn.close()


def worker(job_id: str, pdf_path: str):
    try:
        job_dir = Path(OUTPUTS_DIR) / job_id
        result = extract_pdf_with_ocr(
            job_dir=job_dir,
            pdf_path=Path(pdf_path),
            mode="excel",
            pages="all",
        )
        # Store extraction details as JSON
        import json
        details = json.dumps({
            "tables_extracted": result.get("tables", 0),
            "tables_dropped": result.get("tables_dropped", 0),
            "drop_reasons": result.get("drop_reasons", {})
        })
        update_status(job_id, "verified", excel_path=result["result_path"], details=details)
    except Exception as e:
        update_status(job_id, "rejected", error=str(e))


@app.post("/api/pdf/upload")
async def upload_pdf(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        return JSONResponse({"error": "Only PDF allowed"}, status_code=400)
    job_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOADS_DIR, f"{job_id}_original.pdf")
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    conn = db()
    c = conn.cursor()
    c.execute("INSERT INTO uploads(id, filename, pdf_path, status, created_at) VALUES (?, ?, ?, ?, ?)",
              (job_id, file.filename, pdf_path, "pending", datetime.now().isoformat()))
    conn.commit()
    conn.close()
    if background_tasks:
        background_tasks.add_task(worker, job_id, pdf_path)
    return {"job_id": job_id, "status": "pending", "message": "Processing started"}


@app.get("/api/pdf/jobs/{job_id}/status")
async def status(job_id: str):
    conn = db()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    
    # Parse extraction details if available
    import json
    details = None
    if row["extraction_details"]:
        try:
            details = json.loads(row["extraction_details"])
        except:
            pass
    
    return {
        "job_id": job_id, 
        "status": row["status"], 
        "filename": row["filename"], 
        "error": row["error_msg"],
        "extraction_details": details
    }


@app.get("/api/pdf/jobs/{job_id}/download")
async def download(job_id: str):
    conn = db()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads WHERE id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row or row["status"] != "verified":
        return JSONResponse({"error": "File not ready"}, status_code=400)
    xlsx_path = row["excel_path"]
    if not xlsx_path or not os.path.exists(xlsx_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(xlsx_path, filename=f"{job_id}.xlsx")


@app.get("/api/pdf/uploads")
async def recent():
    conn = db()
    c = conn.cursor()
    c.execute("SELECT * FROM uploads ORDER BY created_at DESC LIMIT 10")
    rows = c.fetchall()
    conn.close()
    return {"uploads": [dict(r) for r in rows]}


# Clustering endpoints
@app.post("/api/clustering/run")
async def clustering_run(
    file: UploadFile = File(...),
    n_clusters: int = 3,
    algo: str = "KMeans"
):
    """Perform clustering on uploaded CSV/Excel file with algorithm selection."""
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
            status_code=400
        )
    
    if not (2 <= n_clusters <= 10):
        return JSONResponse({"error": "n_clusters must be between 2 and 10"}, status_code=400)
    
    if algo not in ["KMeans", "Agglomerative"]:
        return JSONResponse({"error": "algo must be 'KMeans' or 'Agglomerative'"}, status_code=400)
    
    try:
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_DIR) / "clustering" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = job_dir / f"input{file_ext}"
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        output_path = job_dir / "clustered_output.csv"
        metadata = process_clustering_job(input_path, output_path, n_clusters, algo)
        
        return {
            "job_id": job_id,
            "download": f"/api/clustering/download/{job_id}",
            "download_plot": f"/api/clustering/download/{job_id}/plot",
            "plot_base64": metadata["plot_base64"],
            "metadata": {
                "n_clusters": metadata["n_clusters"],
                "n_samples": metadata["n_samples"],
                "algorithm": metadata["algorithm"],
                "columns_used": metadata["columns_used"]
            }
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)


@app.get("/api/clustering/download/{job_id}")
def clustering_download(job_id: str):
    """Download clustered CSV file."""
    output_path = Path(OUTPUTS_DIR) / "clustering" / job_id / "clustered_output.csv"
    if not output_path.exists():
        raise HTTPException(404, "Clustered file not found")
    return FileResponse(str(output_path), media_type="text/csv", filename=f"clustered_{job_id}.csv")


@app.get("/api/clustering/download/{job_id}/plot")
def clustering_download_plot(job_id: str):
    """Download cluster plot PNG file."""
    plot_path = Path(OUTPUTS_DIR) / "clustering" / job_id / "cluster_plot.png"
    if not plot_path.exists():
        raise HTTPException(404, "Cluster plot not found")
    return FileResponse(str(plot_path), media_type="image/png", filename=f"cluster_plot_{job_id}.png")





##PCA
@app.post("/api/pca/run")
async def pca_run(file: UploadFile = File(...), n_components: int = 5):
    """
    Perform PCA on uploaded CSV/Excel file, supporting up to 5 components.
    Initial run returns metadata, default PC1-PC2 plot, and loadings heatmap (base64).
    """
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename).suffix.lower()
    
    # 1. Validation
    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
            status_code=400
        )
    if not (1 <= n_components <= 5):
        return JSONResponse({"error": "n_components must be between 1 and 5"}, status_code=400)
    
    try:
        # 2. Job Setup
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_DIR) / "pca" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Save input file
        input_path = job_dir / f"input{file_ext}"
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # 3. Run Backend Job
        # This function saves all CSVs and the default PC1-PC2 plot and heatmap images.
        metadata = process_pca_job(input_path, job_dir, n_components)
        
        # 4. Return Success Response
        return {
            "job_id": job_id,
            "n_components_run": metadata["n_components"],
            
            # Direct base64 plots for immediate display
            "plot_base64_pc12": metadata["plot_base64_pc12"],
            "heatmap_base64": metadata["heatmap_base64"],
            
            # Download links
            "download_scores": f"/api/pca/download/{job_id}/scores",
            "download_loadings": f"/api/pca/download/{job_id}/loadings",
            "download_variance": f"/api/pca/download/{job_id}/variance",
            "download_heatmap": f"/api/pca/download/{job_id}/heatmap",
            
            # Endpoint template for custom plots (used by frontend)
            "custom_plot_url_template": f"/api/pca/plot/{job_id}/<pc_x>/<pc_y>",
            
            "metadata": {
                "n_components": metadata["n_components"],
                "n_samples": metadata["n_samples"],
                "explained_variance": metadata["explained_variance"],
                "cumulative_variance": metadata["cumulative_variance"],
                "columns_used": metadata["columns_used"]
            }
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        # Catch and handle unexpected errors
        print(f"PCA Processing Error: {e}") 
        return JSONResponse({"error": "Processing failed due to an internal error."}, status_code=500)


@app.get("/api/pca/plot/{job_id}/{pc_x}/{pc_y}")
async def pca_custom_plot(job_id: str, pc_x: int, pc_y: int):
    """
    Generate and return a base64 encoded PCA scatter plot for any two components.
    pc_x and pc_y are 1-indexed component numbers (e.g., 1 for PC1).
    """
    job_dir = Path(OUTPUTS_DIR) / "pca" / job_id
    scores_path = job_dir / "pca_scores.csv"
    variance_path = job_dir / "explained_variance.csv"
    
    # Locate the original input file saved during the run
    input_file_path = next((f for f in job_dir.iterdir() if f.name.startswith("input")), None)

    if not scores_path.exists() or not input_file_path:
        raise HTTPException(404, "PCA job data not found. Run PCA first.")

    try:
        # Load data required for plotting
        scores_df = pd.read_csv(scores_path)
        variance_df = pd.read_csv(variance_path)
        
        # Determine the maximum component index available (0-indexed)
        max_pc_index = variance_df.shape[0] - 1
        
        # Validate requested components (convert 1-indexed to 0-indexed)
        x_idx = pc_x - 1
        y_idx = pc_y - 1
        
        if not (0 <= x_idx <= max_pc_index and 0 <= y_idx <= max_pc_index):
             raise HTTPException(400, f"Invalid component number. Max component available is {max_pc_index + 1}.")
             
        # Extract scores and explained variance as numpy arrays
        # Ensure we only extract the PC columns based on the available components
        pc_columns = [f'PC{i+1}' for i in range(max_pc_index + 1)]
        scores = scores_df[pc_columns].values
        explained_variance = variance_df['Explained_Variance'].values

        # Load original dataframe to retrieve the 'Group' column (for coloring)
        if input_file_path.suffix.lower() == '.csv':
            df_original = pd.read_csv(input_file_path)
        else:
            df_original = pd.read_excel(input_file_path)
            
        # Generate the plot using the backend function
        plot_base64 = create_pca_plot(scores, explained_variance, df_original, pc_x=x_idx, pc_y=y_idx)

        # Return the base64 string
        return {
            "job_id": job_id,
            "pc_x": pc_x,
            "pc_y": pc_y,
            "plot_base64": plot_base64
        }
    except Exception as e:
        print(f"Custom Plot Error: {e}")
        raise HTTPException(500, "Error generating custom plot.")


# --- DOWNLOAD ENDPOINTS ---

# Helper function for downloads to reduce repetition
def get_file_response(job_id: str, filename: str, media_type: str):
    """Generic function to handle file downloads."""
    output_path = Path(OUTPUTS_DIR) / "pca" / job_id / filename
    if not output_path.exists():
        raise HTTPException(404, f"{filename} not found for job ID {job_id}")
    return FileResponse(str(output_path), media_type=media_type, filename=f"{filename.replace('.', '_')}_{job_id}.{media_type.split('/')[-1]}")

@app.get("/api/pca/download/{job_id}/scores")
def pca_download_scores(job_id: str):
    """Download PCA scores CSV."""
    return get_file_response(job_id, "pca_scores.csv", "text/csv")

@app.get("/api/pca/download/{job_id}/loadings")
def pca_download_loadings(job_id: str):
    """Download PCA loadings CSV (Oxide Contributions)."""
    return get_file_response(job_id, "pca_loadings.csv", "text/csv")

@app.get("/api/pca/download/{job_id}/variance")
def pca_download_variance(job_id: str):
    """Download Explained Variance CSV."""
    return get_file_response(job_id, "explained_variance.csv", "text/csv")

@app.get("/api/pca/download/{job_id}/heatmap")
def pca_download_heatmap(job_id: str):
    """Download PCA Loadings Heatmap PNG."""
    return get_file_response(job_id, "pca_heatmap_loadings.png", "image/png")





## TERNARY PLOTTING ENDPOINTS

@app.post("/api/plotting/run")
async def plotting_run(
    file: UploadFile = File(...),
    mode: str = Form("auto"),                    # FIX 1: Use Form()
    system: Optional[str] = Form(None),         # FIX 1: Use Form()
    include_k2o: bool = Form(True)              # FIX 1: Use Form() for boolean
):
    """Perform ternary plotting on uploaded CSV/Excel file."""
    allowed_extensions = {".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename).suffix.lower()
    
    # 1. Validation
    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"error": f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"},
            status_code=400
        )
    if mode not in ["auto", "manual"]:
        return JSONResponse({"error": "mode must be 'auto' or 'manual'"}, status_code=400)
    if mode == "manual" and (not system or system not in TERNARY_MAP):
        return JSONResponse(
            {"error": f"system must be one of: {', '.join(TERNARY_MAP.keys())}"},
            status_code=400
        )
    
    # 2. Job Setup
    try:
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_DIR) / "plotting" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = job_dir / f"input{file_ext}"
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # 3. Run Backend Job
        metadata = process_plotting_job(
            input_path, 
            job_dir, 
            mode=mode,
            system=system,
            include_k2o=include_k2o, # Passes the correctly parsed boolean
            base_dir_root=BASE_DIR 
        )
        
        # 4. Return Success Response
        return {
            "job_id": job_id,
            "download_plot": f"/api/plotting/download/{job_id}/plot",
            "download_data": f"/api/plotting/download/{job_id}/data",
            "plot_base64": metadata["plot_base64"],
            "normalized_data": metadata["normalized_data"],
            "metadata": {
                "system": metadata["system"],
                "n_samples": metadata["n_samples"],
                "mode": metadata["mode"],
                "include_k2o": metadata["include_k2o"], # Reflects the choice
                "corner_names": metadata["corner_names"]
            }
        }
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        print(f"Ternary Plotting Error: {e}")
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)


@app.get("/api/plotting/download/{job_id}/plot")
def plotting_download_plot(job_id: str):
    """Download ternary plot PNG."""
    plot_path = Path(OUTPUTS_DIR) / "plotting" / job_id / "ternary_plot.png"
    if not plot_path.exists():
        raise HTTPException(404, "Ternary plot not found")
    return FileResponse(str(plot_path), media_type="image/png", filename=f"ternary_plot_{job_id}.png")


@app.get("/api/plotting/download/{job_id}/data")
def plotting_download_data(job_id: str):
    """Download normalized data CSV."""
    data_path = Path(OUTPUTS_DIR) / "plotting" / job_id / "normalized_data.csv"
    if not data_path.exists():
        raise HTTPException(404, "Normalized data not found")
    return FileResponse(str(data_path), media_type="text/csv", filename=f"normalized_data_{job_id}.csv")


@app.get("/api/plotting/systems")
def plotting_systems():
    """Return available ternary systems."""
    return {"systems": list(TERNARY_MAP.keys())}




# IMG Anlysis endpoin

@app.post("/api/image-analysis/run")
async def image_analysis_run(
    file: UploadFile = File(...),
    # Parameters are typed as string here because FastAPI might receive them as strings 
    # from FormData, even with the default int hint.
    blur_strength: str = Form(3), 
    threshold_value: str = Form(139), 
    fill_hole_thresh: str = Form(200), 
    min_area: str = Form(50)
):
    """
    Receives image and parameters, runs OpenCV analysis, and returns results.
    The corrected version explicitly handles string-to-integer conversion.
    """
    allowed_types = {"image/jpeg", "image/png", "image/jfif"}
    
    if file.content_type not in allowed_types:
        return JSONResponse(
            {"error": "Invalid file type. Only JPG/JPEG, PNG, and JFIF are supported."},
            status_code=400
        )

    # --- Parameter Validation and Type Conversion ---
    # This block is the crucial fix for the 'str' and 'str' division error.
    try:
        # Explicitly convert form inputs (which come as strings) to integers
        bs = int(blur_strength)
        tv = int(threshold_value)
        fht = int(fill_hole_thresh)
        ma = int(min_area)

        # Basic validation checks using the converted integer variables
        if bs % 2 == 0 or not (1 <= bs <= 11):
            return JSONResponse({"error": "Blur strength must be an odd number between 1 and 11."}, status_code=400)
        if not (0 <= tv <= 255):
            return JSONResponse({"error": "Threshold value must be between 0 and 255."}, status_code=400)

    except ValueError:
        return JSONResponse({"error": "Parameters must be valid integers."}, status_code=400)
    # ---------------------------------------------------

    try:
        job_id = str(uuid.uuid4())
        job_dir = Path(OUTPUTS_DIR) / "image_analysis" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        file_content = await file.read()
        
        # --- Run Analysis: Pass the converted integer variables (bs, tv, fht, ma) ---
        results = analyze_image(
            file_content,
            bs,
            tv,
            fht,
            ma
        )
        
        if results.get("error"):
            return JSONResponse({"error": results["error"]}, status_code=400)

        # --- Return Structured Response ---
        return {
            "job_id": job_id,
            "original_base64": results["original_base64"],
            "result_base64": results["result_base64"],
            "percentage": results["percentage"],
            "ratio": results["ratio"],
            "C_Percent": results["C_Percent"],
            "metadata": { 
                "percentage": results["percentage"],
                "ratio": results["ratio"],
                "C_Percent": results["C_Percent"],
                "parameters_used": {
                    "blur": bs,
                    "threshold": tv,
                    "fill_hole_thresh": fht,
                    "min_area": ma
                }
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # This will catch and log any remaining exceptions
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)
app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static_root")

