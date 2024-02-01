# Use an official PyTorch image as a parent image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 29500 available to the world outside this container
# This port can be used for the PyTorch distributed process group
EXPOSE 29500

# Define environment variable
ENV NAME World
# Define environment variables for distributed training
ENV MASTER_ADDR="localhost"
ENV MASTER_PORT="29500"
ENV NCCL_SOCKET_IFNAME=^docker0,lo

# Run app.py when the container launches
CMD ["python", "functioncallgenerate_mp_mt.py"]