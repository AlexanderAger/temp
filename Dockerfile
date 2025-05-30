# Start with a PyTorch base image that includes CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install additional Python packages
RUN pip install matplotlib tqdm pillow

# First, copy only the requirements to leverage Docker cache
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN mkdir -p /pthfiles

# Copy all project files
COPY VIN.py /app/
COPY run.py /app/
COPY VIN.py /app/
COPY dataset.py /app/
COPY xdt_50l.npz /app/


# Set default command
CMD ["python", "run.py"]
