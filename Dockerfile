# Use the official Python 3.8 image as the base image
FROM python:3.8

# Set the working directory inside the container to /app
WORKDIR /app

# Install wget and unzip to download and extract files
RUN apt-get update && apt-get install -y wget unzip

# Copy the core directory containing other files
COPY core/ core/

# Download the ZIP file from Google Drive
ADD https://drive.google.com/uc?export=download&id=124tuB-Txg1nfv6mp9oc8ZRg7UhzSjC_M /app/core/CMIF/utils/

# Copy the test directory containing other files
COPY test/ test/

# Copy the local requirements.txt file to the container at /app
COPY requirements.txt /app/

# Install dependencies listed in requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the main.py file
COPY main.py .

# Set the entry point to your Python script
ENTRYPOINT ["python", "main.py"]

# Specify default arguments to CMD (can be overridden when running the container)
CMD ["--input_dir", "test/CMIF/data/input", "--output_dir", "test/CMIF/data/output", "--method", "CMIF"]