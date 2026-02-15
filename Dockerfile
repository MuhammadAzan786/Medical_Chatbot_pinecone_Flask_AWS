FROM python:3.10-slim-buster

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

CMD ["python3", "app.py"]