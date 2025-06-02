# Use an official Python image as the base
FROM python:3.11-slim

# Set environment variables to reduce Python output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    tree \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Install sra-tools
RUN wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/current/sratoolkit.current-ubuntu64.tar.gz \
    && tar -xzvf sratoolkit.current-ubuntu64.tar.gz \
    && mkdir -p /usr/local/sra-tools \
    && mv sratoolkit.* /usr/local/sra-tools \
    && ln -s /usr/local/sra-tools/bin/* /usr/local/bin/ \
    && rm -f sratoolkit.current-ubuntu64.tar.gz

# Set the working directory
WORKDIR /app

# Copy the pyproject.toml 
COPY pyproject.toml /app/

# Install the dependencies
RUN uv pip install --system --upgrade pip setuptools wheel \
    && uv pip install --system .

# Copy the source code
COPY SRAgent/ /app/SRAgent/

# Install the package
RUN uv pip install --system .

# Set the default entry point for the container
ENTRYPOINT ["SRAgent"]