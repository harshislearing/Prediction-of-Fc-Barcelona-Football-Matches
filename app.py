from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and preprocessing objects
model = joblib.load('barca_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')
rfe = joblib.load('rfe.pkl')

# Load training data averages (replace with your actual training averages)
training_averages = {
    'POSSESSION': 0.65,
    'SHOTS_ON_TARGET': 5.2,
    'PASS_ACCURACY': 0.85,
    'EXPECTED_GOALS': 1.8,
    'AVG_GOALS_SCORED_LAST_5': 2.1,
    'AVG_GOALS_CONCEDED_LAST_5': 0.9,
    'GOALS_DIFF': 0.3,
    'POSSESSION_SHOTS': 3.25,
    'WIN_STREAK': 1.2,
    'LOSS_STREAK': 0.4
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/club')
def club():
    return render_template('club.html')

@app.route('/visualization')
def visualization():
    return render_template('visualization.html')

@app.route('/prediction')
def prediction():
    return render_template('predictiton.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from frontend
        data = request.json
        opponent_name = data['opponent_name']
        home_or_away = data['home_or_away']

        # Encode opponent
        try:
            opponent_encoded = label_encoder.transform([opponent_name])[0]
        except ValueError:
            return jsonify({'error': f"Opponent '{opponent_name}' not found in the dataset. Please enter a valid opponent name."})

        # Encode home/away
        home_or_away_encoded = 0 if home_or_away.lower() == 'home' else 1

        # Create input data with average values for other features
        input_data = [
            home_or_away_encoded,
            opponent_encoded,
            training_averages['POSSESSION'],
            training_averages['SHOTS_ON_TARGET'],
            training_averages['PASS_ACCURACY'],
            training_averages['EXPECTED_GOALS'],
            training_averages['AVG_GOALS_SCORED_LAST_5'],
            training_averages['AVG_GOALS_CONCEDED_LAST_5'],
            training_averages['GOALS_DIFF'],
            training_averages['POSSESSION_SHOTS'],
            training_averages['WIN_STREAK'],
            training_averages['LOSS_STREAK']
        ]

        # Convert to DataFrame with correct feature names
        feature_names = [
            'HOME_OR_AWAY', 'OPPONENT', 'POSSESSION', 'SHOTS_ON_TARGET',
            'PASS_ACCURACY', 'EXPECTED_GOALS', 'AVG_GOALS_SCORED_LAST_5',
            'AVG_GOALS_CONCEDED_LAST_5', 'GOALS_DIFF', 'POSSESSION_SHOTS',
            'WIN_STREAK', 'LOSS_STREAK'
        ]
        
        input_df = pd.DataFrame([input_data], columns=feature_names)

        # Preprocess the input data
        input_scaled = scaler.transform(input_df)
        input_rfe = rfe.transform(input_scaled)

        # Make prediction
        probabilities = model.predict_proba(input_rfe)[0]

        # Apply away match adjustments as in your Jupyter notebook
        if home_or_away.lower() == 'away':
            win_adj = probabilities[0] * 0.72  # Reduce win probability
            draw_adj = probabilities[1] * 1.18  # Increase draw probability
            loss_adj = probabilities[2] * 1.10  # Increase loss probability
            
            # Normalize to ensure probabilities sum to 1
            total = win_adj + draw_adj + loss_adj
            win_prob = win_adj / total
            draw_prob = draw_adj / total
            loss_prob = loss_adj / total
        else:
            win_prob = probabilities[0]
            draw_prob = probabilities[1]
            loss_prob = probabilities[2]

        return jsonify({
            'Win': float(win_prob),
            'Draw': float(draw_prob),
            'Loss': float(loss_prob)
        })

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)