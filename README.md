# speedEST
Vehicle impact speed estimation using machine learning

# Demonstration

For the purpose of demonstration, the app is deployed 
[here](http://speedest.pl)

# Installation

Build the Docker container with

`docker build -t speedest -f Dockerfile .`

Run the Docker container with

`docker run -p 80:8501 speedest:latest`

The app can then be accessed at 
[localhost](http://localhost)