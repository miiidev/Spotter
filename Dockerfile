# Use the official Python slim image as a base
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir $(grep -v -E 'torch|torchvision|torchaudio' requirements.txt)
COPY . .
EXPOSE 7860
CMD ["python", "app.py"]