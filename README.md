⚽ Prediction of FC Barcelona Football Matches


📋 Project Overview

This project aims to predict the outcomes of FC Barcelona football matches — Win, Draw, or Loss — using machine learning models trained on a self-created dataset. The application is built using Python, Flask, and Scikit-learn and provides predictions based on user inputs such as the opponent and match location (Home/Away).
The model considers several football-specific features like possession, shots on target, pass accuracy, expected goals, and more to make outcome predictions.

🛠️ Technologies Used

Python

Flask (Web Framework)

Scikit-learn (Machine Learning)

Pandas, NumPy (Data Processing)

Joblib (Model Serialization)


📊 Dataset

A self-made dataset (barca_newdataset.csv) was created with the following features:

HOME_OR_AWAY: Match location (0 for Home, 1 for Away)

OPPONENT: Encoded opponent team name

POSSESSION: Ball possession percentage

SHOTS_ON_TARGET: Number of shots on targe

PASS_ACCURACY: Passing accuracy percentage

EXPECTED_GOALS: Expected goals metric

AVG_GOALS_SCORED_LAST_5: Average goals scored in the last 5 matches

AVG_GOALS_CONCEDED_LAST_5: Average goals conceded in the last 5 matches

GOALS_DIFF: Goal difference

POSSESSION_SHOTS: Shots per possession metric

WIN_STREAK: Current win streak

LOSS_STREAK: Current loss streak

🔍 Model Details

The model was trained using feature selection techniques (RFE - Recursive Feature Elimination) and scaling (Standard Scaler). Opponents were label-encoded. The machine learning algorithm was likely a classification model (details not explicitly provided but commonly RandomForest/LogisticRegression).

The .pkl files include:

barca_model.pkl: Trained machine learning model

scaler.pkl: Scaler object for feature normalization

label_encoder.pkl: Label encoder for opponent teams

rfe.pkl: Feature selection object (RFE)

training_averages.pkl: Dictionary of feature averages for input estimation


🖥️ Application Structure

app.py: Main Flask application (API & routes)

/templates: HTML templates for the web interface (home.html, club.html, visualization.html, prediction.html)

.pkl Files: Pre-trained model and preprocessing tools

barca_newdataset.csv: Custom dataset


🧑‍💻 How to Use

Run the Flask app.

Visit the /prediction page.

Provide:

Opponent name

Home or Away

The app will return the probabilities of Win / Draw / Loss.

