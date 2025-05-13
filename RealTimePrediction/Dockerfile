FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make sure the models directory exists
RUN mkdir -p models

EXPOSE 5000

# Run the application with gunicorn for better performance
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "RealTimePrediction:app"]