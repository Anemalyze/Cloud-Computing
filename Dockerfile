# Gunakan image Python sebagai base
FROM python:3.9-slim

# Atur working directory
WORKDIR /app

# Copy requirements dan install dependency
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model file
COPY anemalyze_model.h5 /anemalyze_model.h5

# Copy semua file aplikasi ke kontainer
COPY . .

# Ekspos port 8080
EXPOSE 8080

# Jalankan aplikasi
CMD exec gunicorn -b 0.0.0.0:${PORT:-8080} app:app

