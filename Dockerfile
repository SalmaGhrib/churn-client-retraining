FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .
COPY dataset.csv .
ENV DATASET_PATH=/app/dataset.csv
ENV TARGET_COLUMN=churn
ENV MODEL_OUTPUT=/app/model.pkl
CMD ["python", "train.py"]
