# # Use an official Python runtime as a parent image
# FROM python:3.9-slim

# # Set the working directory in the container
# WORKDIR /app

# # Copy the current directory contents into the container at /app
# COPY . /app

# # Install any needed packages specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Expose port 8080 (Cloud Run expects this)
# ENV PORT 8080
# EXPOSE 8080

# # Run app.py when the container launches
# CMD ["python", "app.py"]


# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy all files from current directory to container
COPY . /app

# Explicitly copy the pdfs folder (ensure it exists in repo)
COPY pdfs /app/pdfs

# List files to confirm PDF exists (for debugging)
RUN ls -la /app/pdfs

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment and expose port
ENV PORT 8000
EXPOSE 8000

# Start the app
CMD ["python", "app.py"]
