from typing import Tuple, Union, Any

from requests import get
from fastapi import FastAPI, Form
from fastapi.staticfiles import StaticFiles
import httpx
import json as _json
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
try:
    import requests_unixsocket  # for Docker API via /var/run/docker.sock
    import requests as _requests
except Exception:
    requests_unixsocket = None
    _requests = None

app = FastAPI()

# Mount the images directory to serve static files
import os
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
images_dir = os.path.join(os.path.dirname(__file__), "images")
app.mount("/images", StaticFiles(directory=images_dir), name="images")

FHIR_URL = os.getenv("FHIR_URL", "http://localhost:8080/fhir")
client = httpx.Client(timeout=300.0)  # 60 seconds
origins = [
    "https://4200-firebase-pancreas-dss-1770494409240.cluster-cbeiita7rbe7iuwhvjs5zww2i4.cloudworkstations.dev"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Only allows requests from this specific URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],    # Still needed for the ngrok-skip-browser-warning header
)
# Utilities to run commands inside the pyradiomics container via Docker API
def _docker_session():
    if requests_unixsocket is None:
        raise RuntimeError("requests-unixsocket not available; install requests-unixsocket.")
    return requests_unixsocket.Session()

def _docker_url(path: str) -> str:
    # http+unix scheme with encoded socket path
    return f"http+unix://%2Fvar%2Frun%2Fdocker.sock{path}"

# No need to search for the container; we can exec by the compose service name

def _exec_in_container(session, container_id_or_name: str, cmd: str) -> str:
    create = session.post(
        _docker_url(f"/containers/{container_id_or_name}/exec"),
        json={
            "AttachStdout": True,
            "AttachStderr": True,
            "Tty": False,
            "Cmd": ["bash", "-lc", cmd],
        },
    )
    create.raise_for_status()
    exec_id = create.json()["Id"]
    start = session.post(
        _docker_url(f"/exec/{exec_id}/start"),
        data=_json.dumps({"Detach": False, "Tty": False}),
        headers={"Content-Type": "application/json"},
    )
    start.raise_for_status()
    # Output may be a raw stream; decode to text
    try:
        return start.text
    except Exception:
        return str(start.content)

def _build_pyradiomics_cmd(nrrd_image_filename: str, nrrd_mask_filename: str, out_csv_name: str, out_dir_name: str) -> str:
    return (
        f"pyradiomics --verbosity {int(os.getenv('PYRADIO_VERBOSITY', '5'))} "
        f"/images/{nrrd_image_filename} /images/{nrrd_mask_filename} "
        f"--mode voxel --jobs={int(os.getenv('PYRADIO_JOBS', '1'))} "
        f"-o /images/{out_csv_name} -f csv --out-dir /images/{out_dir_name} "
        f"-p /images/params.yaml"
    )

async def _run_pyradiomics(nrrd_image_filename: str, nrrd_mask_filename: str, out_csv_name: str, out_dir_name: str) -> Any:
    """
    Execute PyRadiomics either via docker exec (default) or via an HTTP API.
    Configure with:
      - PYRADIO_MODE=exec|api
      - PYRADIO_API_URL=https://... (required for api)
    """
    mode = os.getenv("PYRADIO_MODE", "exec").strip().lower()
    if mode == "api":
        api_url = os.getenv("PYRADIO_API_URL")
        if not api_url:
            raise RuntimeError("PYRADIO_API_URL is required when PYRADIO_MODE=api")
        payload = {
            "image_path": f"/images/{nrrd_image_filename}",
            "mask_path": f"/images/{nrrd_mask_filename}",
            "output_csv": f"/images/{out_csv_name}",
            "output_dir": f"/images/{out_dir_name}",
            "params_path": "/images/params.yaml",
            "mode": "voxel",
            "jobs": int(os.getenv("PYRADIO_JOBS", "1")),
            "verbosity": int(os.getenv("PYRADIO_VERBOSITY", "5")),
            "format": "csv",
        }
        timeout_seconds = float(os.getenv("PYRADIO_API_TIMEOUT", "120"))
        timeout = httpx.Timeout(timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as api_client:
            resp = await api_client.post(api_url, json=payload)
            if resp.status_code not in (200, 201, 202):
                raise RuntimeError(f"PyRadiomics API failed: {resp.status_code} {resp.text}")
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text}

    session = _docker_session()
    container_name = os.getenv("PYRADIO_CONTAINER", "pyradiomics")
    cmd = _build_pyradiomics_cmd(nrrd_image_filename, nrrd_mask_filename, out_csv_name, out_dir_name)
    return _exec_in_container(session, container_name, cmd)

@app.get("/")
async def read_root() -> Union[dict, str]:

    return {"message": "Hello, World!" + os.getenv("FHIR_URL")}

import httpx

@app.get("/patients")
async def getPatients() -> dict:
    print("Fetching ALL patients from FHIR server at:", FHIR_URL)
    
    all_patients = []
    next_url = f"{FHIR_URL}/Patient"
    
    try:
        # extend timeout since fetching all pages might take time
        timeout = httpx.Timeout(300.0) 
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            while next_url:
                print(f"Fetching page: {next_url}")
                response = await client.get(next_url)
                
                if response.status_code != 200:
                    print(f"Error fetching page: {response.status_code}")
                    break
                
                data = response.json()
                
                # 1. Extract patients from the current page's 'entry' list
                #    We grab 'resource' to get the actual Patient object, skipping the wrapper
                if "entry" in data:
                    entries = data["entry"]
                    for entry in entries:
                        if "resource" in entry:
                            all_patients.append(entry["resource"])
                
                # 2. Find the 'next' link for pagination
                next_url = None
                links = data.get("link", [])
                for link in links:
                    if link.get("relation") == "next":
                        next_url = link.get("url")
                        break
                        
    except Exception as e:
        return {"error": f"Failed to fetch patients: {e}"}

    print(f"Total patients fetched: {len(all_patients)}")
    return {"patients": all_patients}
async def log_request(request):
    print(f"Request event hook: {request.method} {request.url} - Waiting for response")
    print("Headers:", request.headers)
@app.get("/addPatient")
async def add_patient():
    """
    Creates a new Patient resource on the FHIR HAPI server with example data.
    """
    patient_resource = {
        "resourceType": "Patient",
        "name": [
            {
                "use": "official",
                "family": "Beverly",
                "given": ["Patrick"]
            }
        ],
        "gender": "male",
        "birthDate": "1965-01-01"
    }
    print(patient_resource)
    print("AINTE REEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(event_hooks={'request': [log_request]},timeout=timeout) as client:
        response = await client.post(f"{FHIR_URL}/Patient", json=patient_resource)
        if response.status_code in (200, 201):
            return {"message": "Patient created", "fhir_response": response.json()}
        else:
            return {"error": "Failed to create patient", "status_code": response.status_code, "details": response.text}

import os
from fastapi import UploadFile, File, HTTPException, Body
import json
from typing import Optional

# optional/soft imports so editor/runtime won't fail at import time; functions will raise if actually needed
import importlib

np = None
cv2 = None
Image = None
tf = None
featureextractor = None
genai = None
genai_types = None

# perform dynamic imports to avoid static "could not be resolved" warnings in editors
# modules will be None if not available at runtime
try:
    _np = importlib.import_module("numpy")
    np = _np
except Exception:
    np = None

try:
    _cv2 = importlib.import_module("cv2")
    cv2 = _cv2
except Exception:
    cv2 = None

try:
    # Import the actual Image module so Image.open is available
    Image = importlib.import_module("PIL.Image")
except Exception:
    Image = None

try:
    _tf = importlib.import_module("tensorflow")
    tf = _tf
except Exception:
    tf = None

try:
    _rad = importlib.import_module("radiomics")
    featureextractor = getattr(_rad, "featureextractor", None) or _rad
except Exception:
    featureextractor = None

try:
    from google import genai as _genai
    from google.genai import types as _genai_types
    genai = _genai
    genai_types = _genai_types
except Exception:
    genai = None
    genai_types = None

try:
    import SimpleITK as sitk
except Exception:
    sitk = None

def build_model(img_size: Tuple[int, int], channels: int = 1) -> Any:
    # Grayscale single-channel input by default
    print("Building CNN model", img_size, "channels", channels)
    inputs = tf.keras.Input(shape=(*img_size, channels), name="input_image")
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="gradcam_cnn")


def _load_weights_grayscale(model, model_path: str) -> None:
    """
    Load weights for a grayscale (1-channel) model.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required to load weights; install tensorflow.")
    try:
        model.load_weights(model_path)
        return
    except ValueError:
        # If weights were trained on RGB, adapt first conv kernel by averaging channels
        rgb_model = build_model((128, 128), channels=3)
        rgb_model.load_weights(model_path)
        rgb_layers = {layer.name: layer for layer in rgb_model.layers}
        for layer in model.layers:
            if not layer.weights:
                continue
            rgb_layer = rgb_layers.get(layer.name)
            if rgb_layer is None or not rgb_layer.weights:
                continue
            rgb_weights = rgb_layer.get_weights()
            if not rgb_weights:
                continue
            if isinstance(layer, tf.keras.layers.Conv2D):
                kernel = rgb_weights[0]
                if kernel.ndim == 4 and kernel.shape[2] == 3 and layer.input_shape[-1] == 1:
                    rgb_weights[0] = kernel.mean(axis=2, keepdims=True)
            layer.set_weights(rgb_weights)


def preprocess_image(image_path):
    """
    Load image, compute radiomic 2D feature map (energy map), resize to 128x128.
    Returns the preprocessed array and the feature map.
    """
    if Image is None:
        raise RuntimeError("PIL.Image is required; install pillow.")
    if np is None:
        raise RuntimeError("NumPy is required; install numpy.")
    # Convert to grayscale and resize
    img = Image.open(image_path).convert("L")
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32)  # shape (128,128), values 0..255
    arr = np.expand_dims(arr, axis=-1)  # shape (128,128,1)
    # Simple 2D feature map for response visibility
    feature_map = np.squeeze(arr, axis=-1)
    return arr, feature_map

def infer_with_cnn(img_array, model_path):
    """
    Load CNN model and perform inference.
    Returns model and probability.
    """
    model = build_model((128, 128), channels=1)
    _load_weights_grayscale(model, model_path)
    prediction = model.predict({"input_image": np.expand_dims(img_array, 0)})
    probability = float(prediction[0][0])
    print(f"Inference probability: {probability}")
    return model, probability
async def async_check_input_validity(image_bytes):
    print("Checking input validity with Gemini SDK...")
    if genai is None or genai_types is None or not os.getenv("GEMINI_API_KEY"):
        return {"explanation": "Gemini SDK not configured (set GEMINI_API_KEY and install google-genai)."}

    try:
        client = genai.Client()
        model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        prompt = os.getenv("GEMINI_PROMPT", "You are working in a hospital as a screener. You are given from the doctor a 2D CT slice image and you have to check if the image is actual a CT or irrelevant. If the image is a CT slice you answer 'valid', if the image is not a CT slice you answer 'invalid'.Answer with just one word, either 'valid' or 'invalid'.")
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )
        print("Gemini SDK response:", response)
        # Try to extract the text field from the response (Gemini SDK may return .text or .candidates[0].content.parts[0].text)
        text = getattr(response, "text", None)
        if not text and hasattr(response, "candidates") and response.candidates:
            # Try to extract from candidates
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts") and candidate.content.parts:
                part = candidate.content.parts[0]
                text = getattr(part, "text", None)
        if not text:
            text = str(response)
        text = (text or "").strip().lower()
        validity = "valid" if "valid" in text and "invalid" not in text else "invalid"
        return {"validity": validity, "gemini_response": text}
    except Exception as e:
        return {"validity": "error", "gemini_response": f"Failed to contact Gemini SDK: {e}"}
def calculate_gradcam(model, img_array, class_index):
    """
    Apply GradCAM using the last Conv2D activations and return the overlay only.
    """
    if tf is None:
        raise RuntimeError("TensorFlow is required for GradCAM; install tensorflow.")
    if np is None:
        raise RuntimeError("NumPy is required for GradCAM; install numpy.")
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for GradCAM; install opencv-python.")

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    img_tensor = tf.expand_dims(img_tensor, 0)

    # Find the last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise RuntimeError("No Conv2D layer found in model for GradCAM.")

    # Build a model that maps the input image to the activations of the last conv layer
    # and the final predictions
    grad_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    # Compute gradients of the top predicted class with respect to the conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    # Global-average-pool the gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv outputs by the pooled grads
    conv_outputs = conv_outputs[0]  # remove batch dim -> (H, W, C)
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.nn.relu(heatmap)
    # Normalize
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to input size and apply colormap
    heatmap = cv2.resize(heatmap, (128, 128))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Build overlay on original grayscale slice
    base_img = img_array.astype(np.uint8)
    if base_img.ndim == 3 and base_img.shape[-1] == 1:
        base_img = np.squeeze(base_img, axis=-1)
    if base_img.ndim == 2:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

    overlay = cv2.addWeighted(base_img, 0.6, heatmap_color, 0.4, 0)
    return overlay

async def send_image_to_gemini(image_path):
    """
    Send GradCAM heatmap image to Gemini for explanation.
    Uses Google GenAI SDK only.
    """
    if genai is None or genai_types is None or not os.getenv("GEMINI_API_KEY"):
        return {"explanation": "Gemini SDK not configured (set GEMINI_API_KEY and install google-genai)."}

    try:
        client = genai.Client()
        model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
        prompt = os.getenv("GEMINI_PROMPT", "You are working in a hospital as a decision support agent. You are given from the doctor an overlayed GRADCAM 2D image.Where the overlay is a heatmap from blue to red indicating  areas that have been considered as important for the model to make the prediction. Provide a trustworthy ethical explanation to the doctor of what he/she is seeing on the overlayed GRADCAM image. Keep the explanation concise and relevant to medical imaging context. You answer should be brief ethical and understandable, no more than 200 words.")
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        response = client.models.generate_content(
            model=model_name,
            contents=[
                prompt,
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            ],
        )
        print("Gemini SDK response:", response)
        return {"explanation": getattr(response, "text", None) or str(response)}
    except Exception as e:
        return {"explanation": f"Failed to contact Gemini SDK: {e}"}


async def get_gemini_explanation(image_path):
    """
    Wrapper kept for compatibility.
    """
    return await send_image_to_gemini(image_path)


async def _save_upload_to_images_dir(image_file: UploadFile) -> str:
    """Save uploaded file to images directory and return absolute path."""
    temp_image_path = os.path.join(images_dir, image_file.filename)
    with open(temp_image_path, "wb") as img_file:
        img_file.write(await image_file.read())
    await image_file.close()
    return temp_image_path


def _prepare_mask_image(image_path: str):
    """Load image and return a grayscale PIL Image for mask creation."""
    if Image is None:
        raise RuntimeError("PIL is required to create mask; install pillow.")
    with Image.open(image_path) as _img:
        return _img.convert("L").copy()


def _write_nrrd_pair(base_img, base_stem: str) -> Tuple[str, str, str, str]:
    """Create image+mask NRRD files from PIL image and return filenames and paths."""
    if sitk is None:
        raise RuntimeError("SimpleITK is required to write NRRD files; install SimpleITK.")
    if np is None:
        raise RuntimeError("NumPy is required for array conversion; install numpy.")

    nrrd_image_filename = f"{base_stem}.nrrd"
    nrrd_mask_filename = f"{base_stem}_mask.nrrd"
    nrrd_image_path = os.path.join(images_dir, nrrd_image_filename)
    nrrd_mask_path = os.path.join(images_dir, nrrd_mask_filename)

    image_array = np.array(base_img)
    mask_array = np.ones_like(image_array, dtype=np.uint8)
    image_itk = sitk.GetImageFromArray(image_array)
    mask_itk = sitk.GetImageFromArray(mask_array)
    mask_itk = sitk.Cast(mask_itk, sitk.sitkUInt8)
    sitk.WriteImage(image_itk, nrrd_image_path)
    sitk.WriteImage(mask_itk, nrrd_mask_path)

    return nrrd_image_filename, nrrd_mask_filename, nrrd_image_path, nrrd_mask_path


def _resolve_model_path() -> str:
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "weights_v1original_gldm_HighGrayLevelEmphasis_run10.weights.h5",
        )
        return model_path

    # Normalize separators to support Windows-style paths in containers
    model_path = model_path.replace("\\", os.sep)

    if os.path.isabs(model_path):
        return model_path

    # Resolve relative paths against repo root and back/ dir
    back_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(back_dir)
    normalized = model_path
    if normalized.startswith("back/") or normalized.startswith("back\\"):
        normalized = normalized.split("back", 1)[-1].lstrip("/\\")
    candidates = [
        os.path.normpath(os.path.join(repo_root, model_path)),
        os.path.normpath(os.path.join(back_dir, model_path)),
        os.path.normpath(os.path.join(back_dir, normalized)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    # Fall back to repo-root resolution
    return candidates[0]

def uuid4() -> str:
    """Generate a random UUID4 string."""
    import uuid
    return str(uuid.uuid4())
def _save_gradcam_image(gradcam_img, original_filename: str, suffix: str = "") -> Tuple[str, str]:
    base_stem = os.path.splitext(original_filename)[0]
    name_suffix = f"_{suffix}" if suffix else ""
    gradcam_filename = f"gradcam_{uuid4()}.png"
    gradcam_path = os.path.join(images_dir, gradcam_filename)
    cv2.imwrite(gradcam_path, gradcam_img)
    return gradcam_filename, gradcam_path


def _ensure_radiomics_output_dir(base_stem: str) -> Tuple[str, str]:
    out_dir_name = f"voxel_out_{base_stem}"
    os.makedirs(os.path.join(images_dir, out_dir_name), exist_ok=True)
    out_csv_name = f"results_voxel_{base_stem}.csv"
    return out_dir_name, out_csv_name


def _cleanup_paths(*paths: str) -> None:
    import shutil
    for path in paths:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        except PermissionError:
            pass
        except FileNotFoundError:
            pass
        except Exception:
            pass

# (removed duplicate docker_exec and endpoint; pyradiomics will be invoked within gemini_assessment)

def prepare_fhir_observation(probability, gradcam_path, radiomic_features, explanation, patient_ref="Patient/example"):
    """
    Prepare the FHIR Observation dict.
    """
    return {
        "resourceType": "Observation",
        "status": "final",
        "code": {
            "coding": [{
                "system": "http://loinc.org",
                "code": "12345-6",
                "display": "CT Slice Assessment"
            }]
        },
        "subject": {"reference": patient_ref},
        "valueQuantity": {
            "value": probability,
            "unit": "probability",
            "system": "http://unitsofmeasure.org",
            "code": "1"
        },
        "component": [
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "12345-7",
                            "display": "Radiomic Features"
                        }
                    ]
                },
                "valueString": json.dumps(radiomic_features)
            },
            {
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": "12345-8",
                            "display": "GradCAM Path"
                        }
                    ]
                },
                "valueString": gradcam_path
            }
        ],
        "note": [{"text": explanation}]
    }

@app.post("/gemini/assessment/")
async def gemini_assessment(
    image_file: UploadFile = File(...),
    patient_id: str = Form(...)
) -> dict:
    """
    Processes a raw CT slice and mask: extracts radiomic features using PyRadiomics container, resizes, infers with CNN,
    applies GradCAM, stores GradCAM image, calls Gemini for explanation, stores assessment as FHIR Observation.
    """
    # Read image bytes directly from UploadFile
    image_bytes = await image_file.read()
    print("request t gemini on truck")
    validity_check = await async_check_input_validity(image_bytes)
    print("Gemini validity check:", validity_check)
    print("Gemini response text:", validity_check.get("gemini_response"))
    if validity_check.get("validity") == "invalid":
        # _cleanup_paths(image_file.file)  # ensure file is closed and cleaned up
        return {"error": "Uploaded image is not a valid CT slice.", "gemini_response": validity_check.get("gemini_response")}
    # Reset file pointer for further use
    image_file.file.seek(0)
    temp_image_path = await _save_upload_to_images_dir(image_file)

    # Ensure mask is full-image labelmap (all ones) created internally
    base_img = _prepare_mask_image(temp_image_path)

    # Convert PNGs to temporary NRRD files for PyRadiomics CLI
    base_stem = os.path.splitext(image_file.filename)[0]
    (
        nrrd_image_filename,
        nrrd_mask_filename,
        nrrd_image_path,
        nrrd_mask_path,
    ) = _write_nrrd_pair(base_img, base_stem)
    print(f"Saved NRRD image to {nrrd_image_path} and mask to {nrrd_mask_path}")
    # Preprocess and extract radiomic 2D feature map
    img_array, radiomic_feature_map = preprocess_image(temp_image_path)

    # Input to CNN for inference
    model_path = _resolve_model_path()
    if not os.path.exists(model_path):
        return {"error": f"Model weights not found at {model_path}"}
    model, probability = infer_with_cnn(img_array, model_path)

    # Calculate GradCAM overlay
    gradcam_overlay = calculate_gradcam(model, img_array, 0)

    # Save overlayed GradCAM image (used for UI and Gemini)
    overlay_filename, overlay_path = _save_gradcam_image(gradcam_overlay, image_file.filename, "overlay")

    # Run pyradiomics voxel extraction in container with direct image+mask paths
    out_dir_name, out_csv_name = _ensure_radiomics_output_dir(base_stem)

    try:
        _ = await _run_pyradiomics(nrrd_image_filename, nrrd_mask_filename, out_csv_name, out_dir_name)
    except Exception as e:
        return {"error": f"Failed to run pyradiomics: {e}",
                "status":"error"}

    # Reference the radiomics output directory for downstream use
    radiomic_features = f"/images/{out_dir_name}"

    # Send GradCAM overlay to Gemini for AI explanation
    gemini_response = await get_gemini_explanation(overlay_path)
    explanation = gemini_response.get("explanation", "No explanation available")
    # explanation = "test"
    print(f"Gemini explanation: {explanation}")
    # print("gemini response", gemini_response)
    print("patientId", patient_id)
    # Prepare FHIR object
    observation = prepare_fhir_observation(probability, f"/images/{overlay_filename}", radiomic_features, explanation, patient_ref=f"Patient/{patient_id or 'example'}")

    # Post to FHIR
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{FHIR_URL}/Observation", json=observation)
        if response.status_code not in (200, 201):
            return {"error": "Failed to create observation", "status": "error"}

    # Clean up
    print("What iam cleaning up", temp_image_path, nrrd_image_path, nrrd_mask_path, "/app/images/"+out_csv_name,"/app"+radiomic_features)
    _cleanup_paths(temp_image_path, nrrd_image_path, nrrd_mask_path, "/app/images/"+out_csv_name,"/app"+radiomic_features)
    print({
        "probability": probability,
        "gradcam_path": f"/images/{overlay_filename}",
        "radiomic_features": radiomic_features,
        # include radiomic_feature_map info so the variable is used and visible in responses
        "radiomic_feature_map_shape": getattr(radiomic_feature_map, "shape", None),
        "gemini_explanation": explanation
    })
    
    return {
        "status": "success",
        "result": response.json(),
    }
    
@app.post("/assessments")
async def getPatientAssessments(patient_id: str = Body(...)) -> dict:
    """Fetch all Observations for a given patient from FHIR server.
    """
    timeout = httpx.Timeout(300.0)
    all_observations = []
    next_url = f"{FHIR_URL}/Observation?subject=Patient/{patient_id}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            while next_url:
                response = await client.get(next_url)
                if response.status_code != 200:
                    return {"error": f"Failed to fetch observations for patient {patient_id}", "status": "error"}
                data = response.json()
                # Flatten observations from 'entry' list
                if "entry" in data:
                    entries = data["entry"]
                    for entry in entries:
                        if "resource" in entry:
                            all_observations.append(entry["resource"])
                # Find the 'next' link for pagination
                next_url = None
                links = data.get("link", [])
                for link in links:
                    if link.get("relation") == "next":
                        next_url = link.get("url")
                        break
    except Exception as e:
        return {"error": f"Failed to fetch observations: {e}"}
    return {"observations": all_observations}
        
    # return {
    #     "probability": probability,
    #     "gradcam_path": f"/images/{overlay_filename}",
    #     "radiomic_features": radiomic_features,
    #     # include radiomic_feature_map info so the variable is used and visible in responses
    #     "radiomic_feature_map_shape": getattr(radiomic_feature_map, "shape", None),
    #     "gemini_explanation": explanation,
    #     # "fhir_observation_id": response.json().get("id")
    # }


# (removed endpoint; radiomics batch execution is now part of gemini_assessment pipeline)