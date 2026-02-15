FROM python:3.10-slim-buster

WORKDIR /app

# Copy setup.py and requirements.txt first for better layer caching
COPY requirements.txt setup.py /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

CMD ["python3", "app.py"]