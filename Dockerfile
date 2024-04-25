# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install utilities for network diagnostics
RUN apt-get update && apt-get install -y iputils-ping curl

# Network diagnostics
RUN ping -c 4 google.com
RUN curl -I https://pypi.org/simple/torch/

# Install pip requirements
COPY core/deep_justintime/requirements.txt .
# First install only PyTorch to ensure it's available
RUN python -m pip install torch>=2.1.0
# Then install all other requirements
RUN python -m pip install -r requirements.txt



WORKDIR /app
COPY . .

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]
