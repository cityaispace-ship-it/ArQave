import cv2 as cv
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# def analyze_image(file_content: bytes, blur_strength: int, threshold_value: int, fill_hole_thresh: int, min_area: int) -> dict:
#     """
#     Performs cementite analysis using OpenCV on image content.
#     Returns results and the processed image as a Base64 string.
#     """
#     # Convert file bytes to numpy array for OpenCV
#     file_bytes = np.asarray(bytearray(file_content), dtype=np.uint8)
#     im = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

#     if im is None:
#         raise ValueError("Could not decode image file.")

#     # Save original image as Base64 for return
#     _, buffer = cv.imencode('.png', cv.cvtColor(im, cv.COLOR_BGR2RGB))
#     original_base64 = base64.b64encode(buffer).decode('utf-8')
    
#     # --- Step 1: Fill holes (Simulation) ---
#     imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    
#     # We work on a copy of the image for filling
#     im_fill = im.copy()
#     blur_fill = cv.GaussianBlur(imgray, (1, 1), 0)
#     inv_blur = cv.bitwise_not(blur_fill)
#     _, thresh_fill = cv.threshold(inv_blur, fill_hole_thresh, 255, cv.THRESH_BINARY)
    
#     # Note: Using cv.RETR_EXTERNAL often works better for filling holes if the background is uniform
#     contours_fill, _ = cv.findContours(thresh_fill, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
#     # Draw white filled contours on the copy
#     cv.drawContours(im_fill, contours_fill, -1, (255, 255, 255), cv.FILLED)

#     # --- Step 2: Reprocess image after filling ---
#     imgray_reproc = cv.cvtColor(im_fill, cv.COLOR_BGR2GRAY)
#     # Blur strength must be an odd number (enforced by the slider in the Streamlit original)
#     if blur_strength % 2 == 0:
#         blur_strength += 1
        
#     blur = cv.GaussianBlur(imgray_reproc, (blur_strength, blur_strength), 0)

#     # --- Step 3: Thresholding ---
#     _, thresh = cv.threshold(blur, threshold_value, 255, cv.THRESH_BINARY)

#     # --- Step 4: Find contours and filter small areas ---
#     contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     cementite_pixels = 0
#     valid_contours = []
    
#     # Prepare the result image: copy of the processed grayscale image, converted to BGR
#     result_img_display = cv.cvtColor(imgray_reproc, cv.COLOR_GRAY2BGR) 
    
#     for cnt in contours:
#         area = cv.contourArea(cnt)
#         if area > min_area:
#             valid_contours.append(cnt)
#             # Accumulate cementite area
#             cementite_pixels += area

#     # Draw the valid contours on the result image (in green)
#     cv.drawContours(result_img_display, valid_contours, -1, (0, 255, 0), 2)
    
#     # Convert result image to Base64
#     _, buffer = cv.imencode('.png', cv.cvtColor(result_img_display, cv.COLOR_BGR2RGB))
#     result_base64 = base64.b64encode(buffer).decode('utf-8')

#     # --- Calculations ---
#     total_area = imgray_reproc.shape[0] * imgray_reproc.shape[1]
#     pearlite_pixels = total_area - cementite_pixels

#     if total_area == 0 or pearlite_pixels <= 0: # Check for non-positive pearlite
#         return {
#             "original_base64": original_base64,
#             "result_base64": "",
#             "percentage": 0.0,
#             "ratio": 0.0,
#             "C_Percent": 0.0,
#             "error": "Total area is zero or only cementite was found (Pearlite pixels <= 0)."
#         }

#     percentage = cementite_pixels * 100 / total_area
#     ratio = cementite_pixels / pearlite_pixels
#     # C_Percent = ((6.67 * cementite_pixels) + (0.8 * pearlite_pixels)) / total_area
#     # Assuming the calculation is based on area fraction of phases
#     # 6.67 wt% C in Cementite, 0.8 wt% C in Pearlite (approximated for Ferrite/Alpha-iron)
#     C_Percent = ((6.67 * cementite_pixels) + (0.8 * pearlite_pixels)) / total_area

#     return {
#         "original_base64": original_base64,
#         "result_base64": result_base64,
#         "percentage": percentage,
#         "ratio": ratio,
#         "C_Percent": C_Percent,
#         "error": None
#     }

# # End of img_analysis_service.py


import cv2 as cv
import numpy as np
import base64
import uuid
from pathlib import Path
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# # --- CONFIGURATION ---
# BASE_DIR = Path(__file__).parent
# OUTPUTS_DIR = BASE_DIR / "outputs"
# OUTPUTS_DIR.mkdir(exist_ok=True)

# --- IMAGE ANALYSIS SERVICE LOGIC ---

def analyze_image(file_content: bytes, blur_strength: int, threshold_value: int, fill_hole_thresh: int, min_area: int) -> dict:
    """
    Performs cementite analysis using OpenCV on image content based on parameters.
    Returns results and the processed image as a Base64 string.
    """
    try:
        # Convert file bytes to numpy array for OpenCV
        file_bytes = np.asarray(bytearray(file_content), dtype=np.uint8)
        im = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        if im is None:
            raise ValueError("Could not decode image file. File may be corrupted or format unsupported.")

        # Save original image as Base64 for return
        _, buffer = cv.imencode('.png', cv.cvtColor(im, cv.COLOR_BGR2RGB))
        original_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # --- Step 1: Fill holes (Simulation) ---
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        
        im_fill = im.copy()
        blur_fill = cv.GaussianBlur(imgray, (1, 1), 0)
        inv_blur = cv.bitwise_not(blur_fill)
        _, thresh_fill = cv.threshold(inv_blur, fill_hole_thresh, 255, cv.THRESH_BINARY)
        
        contours_fill, _ = cv.findContours(thresh_fill, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cv.drawContours(im_fill, contours_fill, -1, (255, 255, 255), cv.FILLED)

        # --- Step 2: Reprocess image after filling ---
        imgray_reproc = cv.cvtColor(im_fill, cv.COLOR_BGR2GRAY)
        
        if blur_strength % 2 == 0:
            blur_strength += 1
            
        blur = cv.GaussianBlur(imgray_reproc, (blur_strength, blur_strength), 0)

        # --- Step 3: Thresholding ---
        _, thresh = cv.threshold(blur, threshold_value, 255, cv.THRESH_BINARY)

        # --- Step 4: Find contours and filter small areas ---
        contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cementite_pixels = 0
        valid_contours = []
        
        result_img_display = cv.cvtColor(imgray_reproc, cv.COLOR_GRAY2BGR) 
        
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > min_area:
                valid_contours.append(cnt)
                cementite_pixels += area

        cv.drawContours(result_img_display, valid_contours, -1, (0, 255, 0), 2)
        
        # Convert result image to Base64
        _, buffer = cv.imencode('.png', cv.cvtColor(result_img_display, cv.COLOR_BGR2RGB))
        result_base64 = base64.b64encode(buffer).decode('utf-8')

        # --- Calculations ---
        total_area = imgray_reproc.shape[0] * imgray_reproc.shape[1]
        pearlite_pixels = total_area - cementite_pixels

        if total_area == 0 or pearlite_pixels <= 0:
             return {
                "original_base64": original_base64,
                "result_base64": "",
                "percentage": 0.0,
                "ratio": 0.0,
                "C_Percent": 0.0,
                "error": "Total area is zero or only cementite was found (Pearlite pixels <= 0)."
            }

        percentage = cementite_pixels * 100 / total_area
        ratio = cementite_pixels / pearlite_pixels
        C_Percent = ((6.67 * cementite_pixels) + (0.8 * pearlite_pixels)) / total_area

        return {
            "original_base64": original_base64,
            "result_base64": result_base64,
            "percentage": percentage,
            "ratio": ratio,
            "C_Percent": C_Percent,
            "error": None
        }
    
    except Exception as e:
        # For general errors in analyze_image, use a specific format
        return {"error": f"Analysis failed: {str(e)}", "result_base64": ""}

# --- FASTAPI APP SETUP ---

app = FastAPI()

@app.post("/api/image-analysis/run")
async def image_analysis_run(
    file: UploadFile = File(...),
    blur_strength: str = Form(3), 
    threshold_value: str = Form(139), 
    fill_hole_thresh: str = Form(200), 
    min_area: str = Form(50)
):
    """
    Receives image and parameters, runs OpenCV analysis, and returns results.
    """
    allowed_types = {"image/jpeg", "image/png", "image/jfif"}
    
    if file.content_type not in allowed_types:
        return JSONResponse(
            {"error": "Invalid file type. Only JPG/JPEG, PNG, and JFIF are supported."},
            status_code=400
        )

    # --- Parameter Validation and Type Conversion ---
    # The explicit int() conversion is moved here to ensure parameters are numbers.
    try:
        # Explicitly convert form inputs (which come as strings) to integers
        bs = int(blur_strength)
        tv = int(threshold_value)
        fht = int(fill_hole_thresh)
        ma = int(min_area)

        # Basic validation checks
        if bs % 2 == 0 or not (1 <= bs <= 11):
            return JSONResponse({"error": "Blur strength must be an odd number between 1 and 11."}, status_code=400)
        if not (0 <= tv <= 255):
            return JSONResponse({"error": "Threshold value must be between 0 and 255."}, status_code=400)

    except ValueError:
        return JSONResponse({"error": "Parameters must be valid integers."}, status_code=400)
    # ---------------------------------------------------

    try:
        job_id = str(uuid.uuid4())
        job_dir = OUTPUTS_DIR / "image_analysis" / job_id
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
        return JSONResponse({"error": f"Processing failed: {str(e)}"}, status_code=500)

# # --- STATIC FILES MOUNT ---
# app.mount("/", StaticFiles(directory=BASE_DIR, html=True), name="static_root")