# intensity-analysis-nlp
# Sentiment Analysis Flask API

This project demonstrates a sentiment analysis model deployed as a Flask web application. The model predicts the sentiment of a given text input (angriness, happiness, sadness).

## Project Structure

sentiment-analysis-flask-api/
├── app.py
├── sentiment_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── Procfile
├── README.md
└── .gitignore

Step-by-Step Implementation Plan:
-Data Collection: Load the data from the provided dataset.
-Data Preprocessing: Clean and preprocess the text data.
-Feature Engineering: Extract relevant features from the text.
-Model Selection and Training: Choose and train machine learning models.
-Model Evaluation: Evaluate the models using appropriate metrics.
-Hyperparameter Tuning: Optimize model hyperparameters.
-Model Deployment Plan: Outline steps for deploying the model.
-Documentation and Reporting: Prepare a detailed report and package the solution

Intensity Analysis (Build your own model using NLP and Python) 
The objective of this project is to develop an intelligent system using NLP to predict the intensity in the text reviews. By analyzing various parameters and process data, the system will predict the intensity where its happiness, angriness or sadness. This predictive capability will enable to proactively optimize their processes, and improve overall customer satisfaction.

# Emotion Classification Project

## Overview
This project classifies text into different emotions (angriness, happiness) using a Logistic Regression model. The project involves data preprocessing, model training, and deployment using a Flask API.

## Dataset
The datasets used in this project are:
- Angriness dataset (`angriness.csv`)
- Happiness dataset (`happiness.csv`)
- Sadness dataset (`sadness.csv`)

## Preprocessing
The text data is preprocessed by:
- Converting to lowercase
- Tokenizing
- Removing stopwords
- Lemmatizing

## Model Training
A Logistic Regression model is trained using TfidfVectorizer to vectorize the text data. GridSearchCV is used to find the best hyperparameters.

## Model Deployment
A Flask API is created to serve the model. The API accepts text input and returns the predicted emotion.

## Files
- `train.py`: Script to preprocess data, train the model, and save the model and vectorizer.
- `app.py`: Flask application to serve the model.
- `model.pkl`: Trained model.
- `vectorizer.pkl`: TfidfVectorizer.

## Usage
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
