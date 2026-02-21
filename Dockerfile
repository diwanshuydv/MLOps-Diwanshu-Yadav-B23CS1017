# Use the official PyTorch image (includes Python, PyTorch, and CUDA support)
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install uv, create a virtual environment, and install dependencies using uv
RUN pip install uv && \
    uv venv .venv && \
    . .venv/bin/activate && \
    uv pip install --no-cache -r requirements.txt

# Copy the rest of your application code
COPY train.py evaluate.py setA.pth ./

# Copy the data directory
COPY data ./data

# # Activate the env and then run the cmd
# CMD ["/bin/bash", "-c", "source .venv/bin/activate && python evaluate.py"]

# Activate the env and then run the cmd
CMD ["/bin/bash", "-c", "source .venv/bin/activate && python evaluate.py"]