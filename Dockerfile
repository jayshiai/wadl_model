FROM python:3.8


# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
# Make port 5000 available to the world outside this container
EXPOSE 6000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
