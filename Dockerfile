FROM python:3.8

# Set the working directory in the container
WORKDIR /code

# Install GDAL dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install -y gdal-bin libgdal-dev g++


COPY ./web_neda/requirements.txt /code/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of your application
COPY ./web_neda/app.py /code/app.py

# Run app.py when the container launches
CMD ["python", "app.py"]
