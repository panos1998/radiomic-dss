from typing import Optional
import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="PyRadiomics API")

class PyRadiomicsRequest(BaseModel):
    image_path: str = Field(..., description="Path to image file inside container, e.g. /images/1.nrrd")
    mask_path: str = Field(..., description="Path to mask file inside container, e.g. /images/1_mask.nrrd")
    output_csv: str = Field(..., description="Output CSV file path, e.g. /images/results.csv")
    output_dir: str = Field(..., description="Output directory for voxel mode, e.g. /images/voxel_out")
    params_path: str = Field("/images/params.yaml", description="Path to params.yaml")
    mode: str = Field("voxel", description="Radiomics mode")
    jobs: int = Field(1, description="Parallel jobs")
    verbosity: int = Field(5, description="Verbosity level")
    format: str = Field("csv", description="Output format")

class PyRadiomicsResponse(BaseModel):
    returncode: int
    stdout: str
    stderr: str

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

@app.post("/run", response_model=PyRadiomicsResponse)
async def run_pyradiomics(payload: PyRadiomicsRequest) -> PyRadiomicsResponse:
    cmd = [
        "pyradiomics",
        "--verbosity",
        str(payload.verbosity),
        payload.image_path,
        payload.mask_path,
        "--mode",
        payload.mode,
        f"--jobs={payload.jobs}",
        "-o",
        payload.output_csv,
        "-f",
        payload.format,
        "--out-dir",
        payload.output_dir,
        "-p",
        payload.params_path,
    ]

    try:
        print("receiving payload", payload)
        print("receiving command", cmd)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            cwd=os.getcwd(),
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="pyradiomics CLI not found in container")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run pyradiomics: {exc}")

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            },
        )

    return PyRadiomicsResponse(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
