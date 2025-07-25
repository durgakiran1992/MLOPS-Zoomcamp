FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

## Copy your scoring script into the image
COPY module4-homework.py .

# Install dependencies
RUN pip install pandas pyarrow scikit-learn==1.5.0

# Run the script by default (optional - you can override this in docker run)
ENTRYPOINT ["python", "module4-homework.py"]
