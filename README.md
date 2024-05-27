 ![](src/img/logo/logo_large_light.png)

# Vehicle impact speed estimation using machine learning

This tool can be used to estimate the speed of the vehicle at impact with a steel road safety barrier. 
The estimation is made by a few different machine learning model, which were trained on both full-scale crash tests and numerical simulation.
The app consists of the following parts:

* __Speed Estimator__ - prediction of the impact speed for a given set of input features
* __Data__ - description of the input features and the datasets used in development of machine learning models
* __Tree Ensemble__ - model based on regression trees
* __Regularized Linear Ensemble__ - model based on linear regressors
* __Support Vector Ensemble__ - model based on support vector machines
* __Multilayer Perceptron__ - model based on a neural network
* __Final Voting Ensemble__ - ensemble of machine learning models
* __About the project__ - summary of the research project
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