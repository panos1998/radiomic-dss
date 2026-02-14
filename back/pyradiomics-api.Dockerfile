FROM radiomics/pyradiomics:latest

WORKDIR /app
COPY pyradiomics_api.py /app/pyradiomics_api.py
# Pin to versions compatible with Python 3.9 in radiomics/pyradiomics image
RUN pip install --no-cache-dir fastapi==0.103.2 uvicorn==0.23.2 pydantic==1.10.13

EXPOSE 8001
CMD ["uvicorn", "pyradiomics_api:app", "--host", "0.0.0.0", "--port", "8001", "--reload", "--reload-dir", "/app"]
