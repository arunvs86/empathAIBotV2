# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files from current directory to container
COPY . /app

# Explicitly copy the pdfs folder (ensure it exists in repo)
COPY pdfs /app/pdfs

# Show files for debugging
RUN ls -la /app/pdfs

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment and expose port
ENV PORT 8000
EXPOSE 8000

# Start using gunicorn instead of python directly
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
