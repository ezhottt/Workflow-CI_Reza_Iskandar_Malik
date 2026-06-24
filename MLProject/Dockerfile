FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install pandas scikit-learn mlflow==2.19.0
# Ini akan menjalankan training pas container di-build
CMD ["python", "modelling.py"]