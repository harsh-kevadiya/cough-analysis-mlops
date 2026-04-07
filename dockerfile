# 1. Upgrade to Python 3.11 to support your modern ML requirements
FROM python:3.11-slim

# 2. Re-install system audio dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Upgrade pip immediately to avoid the 'notice' and installation issues
RUN pip install --upgrade pip

# 4. Copy and Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your project
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]