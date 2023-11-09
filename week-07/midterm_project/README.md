# Midterm Project

by: [Balthazar Paixao](https://www.linkedin.com/in/balthapaixao/)

# The Problem - Flght Price Prediction

Every day, thousands of people search for flights to travel around the world. One thing that always comes to mind is the price of the flight. How much will it cost? Is it a good price? Should I buy now or wait for a better price? These are all questions that we ask ourselves when we are looking for a flight. The problem is that there are so many variables that affect the price of a flight that it is almost impossible to predict the price of a flight. This is where machine learning comes in. With machine learning, we can use data from previous flights to predict the price of a flight.

Considering that, this project aims to predict the price of a flight ticket based on the information provided by the user. The data was obtained from Kaggle and it is composed of two datasets, one with the data from business flights and the other with the data from economy flights.

# The Data

The data used in this project is available in the following link: [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

They already pre processed a dataset and make it available for download, but I decided to do it myself to get more familiar with the data and to practice my data cleaning skills.

# EDA and Pre Processing

Aiming to get a better understanding of the data, I did some exploratory data analysis and data cleaning.

## EDA

It all began with the EDA process to understand the variaables, their distribution and their relationship with the target variable. All the steps commented can be found in the notebook [`eda.ipynb`](https://github.com/balthapaixao/ml-zoomcamp/blob/main/week-07/midterm_project/scripts/eda.ipynb).

## Pre Processing

The pre processing has the goal to remove noise from the data andclassify some of thevariables. The process is also commented and can be found in the notebook [`preprocessing.ipynb`](https://github.com/balthapaixao/ml-zoomcamp/blob/main/week-07/midterm_project/scripts/preprocessing.ipynb). The final dataset is saved in the file [`preprocessed_data.csv`](https://github.com/balthapaixao/ml-zoomcamp/blob/main/week-07/midterm_project/data/preprocessed_data.csv).

# Modeling

The modeling development process can also be found in a notebook [`model.ipynb`](https://github.com/balthapaixao/ml-zoomcamp/blob/main/week-07/midterm_project/scripts/model.ipynb).

The main steps for the modeling process are:

- Split the data into train, validation and test sets;
- Train a baseline model for each algorithm
  - Linear Regression;
  - Random Forest;
  - Gradient Boosting;
- Tune the hyperparameters of the best model;
- Evaluate the model on the test set.

# Deployment

The code for the deployment can be found in the [scripts folder](https://github.com/balthapaixao/ml-zoomcamp/tree/main/week-07/midterm_project/scripts).

The dockerfile is used to create a docker image with all the dependencies needed to run the project. The docker image is then used to create a container that runs the project. The container is deployed using guinicorn to keep the app running and receive the POST requests.

# How to run the project

Up the service with docker.

```bash
docker build -t flight_price_prediction .
docker run -it --rm -p 9696:9696 flight_price_prediction
```

In case you need to delete the image and the container.

```bash
docker rmi -f flight_price_prediction
```

Place a POST request to the service.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"date":"15-10-2023","airline":"SpiceJet", "ch_code":"SG", "num_code":"287", "dep_time":"10:10", "arr_time":"12:35", "time_taken":"2h 25m", "stop":"non-stop", "from":"Delhi", "to":"Mumbai", "class":"business"}' http://localhost:9696/predict
```

# Next Steps

Even though it was a bonus point the deployment in the cloud, the very next step will be to deploy the model in the cloud. I will use AWS to deploy the model and create an API to receive the POST requests.

Insert logs in the code to keep track of the requests and the predictions.

# Final comments

Thank you for reading this far. I hope you enjoyed the project and feel free to reach out to me if you have any questions or suggestions.

Contact me on [LinkedIn](https://www.linkedin.com/in/balthapaixao/).
