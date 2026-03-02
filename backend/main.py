from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import cv2
import numpy as np
import io
import json

from processing.filters import FILTERS

app = FastAPI(title="Medical Image Processing API")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Enable CORS for browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------

# @app.get("/")
# def home():
#     return {
#         "status": "running",
#         "service": "Medical Image Processing API"
#     }

@app.get("/")
def serve_ui():
    return FileResponse("frontend/index.html")

# ---------------------------------------------------
# Core Processing Endpoint (Loosely Coupled)
# ---------------------------------------------------

@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    config: str = Form(...)
):
    """
    config JSON example:
    {
        "operation": "gaussian_blur",
        "parameters": {
            "kernel": 7
        }
    }
    """

    # -------------------------
    # Parse config JSON
    # -------------------------
    try:
        config_data = json.loads(config)
        operation = config_data["operation"]
        parameters = config_data.get("parameters", {})
    except Exception as e:
        return JSONResponse(
            {"error": "Invalid config JSON", "detail": str(e)},
            status_code=400
        )

    # -------------------------
    # Validate operation
    # -------------------------
    if operation not in FILTERS:
        return JSONResponse(
            {"error": f"Unsupported operation: {operation}"},
            status_code=400
        )

    # -------------------------
    # Read uploaded image
    # -------------------------
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Image decode failed")

    except Exception as e:
        return JSONResponse(
            {"error": "Invalid image file", "detail": str(e)},
            status_code=400
        )

    # -------------------------
    # Apply processing (PLUGIN CALL)
    # -------------------------
    try:
        result = FILTERS[operation](image, parameters)
    except Exception as e:
        return JSONResponse(
            {"error": "Processing failed", "detail": str(e)},
            status_code=500
        )

    # -------------------------
    # Encode output image
    # -------------------------
    success, buffer = cv2.imencode(".png", result)
    if not success:
        return JSONResponse(
            {"error": "Image encoding failed"},
            status_code=500
        )

    # -------------------------
    # Return processed image
    # -------------------------
    output_name = f"{operation}.png"

    return StreamingResponse(
    io.BytesIO(buffer.tobytes()),
    media_type="image/png",
    headers={
        "X-Operation-Name": operation
    }
)



# ---------------------------------------------------
# List Available Operations (UI uses this later)
# ---------------------------------------------------

@app.get("/operations")
def list_operations():
    return {
        "available_operations": list(FILTERS.keys())
    }
