from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)
CORS(app)  # Enables frontend HTML to communicate with the backend

# ==============================================================================
# MACHINE LEARNING ENGINE & DATA PREPROCESSING
# ==============================================================================
def train_ipl_model():
    print("⏳ Loading CSV datasets...")
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    
    # Standardize team names to merge historical duplicates
    team_mapping = {
        'Delhi Daredevils': 'Delhi Capitals', 
        'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiants': 'Rising Pune Supergiant', 
        'Deccan Chargers': 'Sunrisers Hyderabad'
    }
    for df in [matches, deliveries]:
        df.replace(team_mapping, inplace=True)

    eligible_teams = [
        'Sunrisers Hyderabad', 'Mumbai Indians', 'Kolkata Knight Riders', 
        'Royal Challengers Bangalore', 'Punjab Kings', 'Chennai Super Kings', 
        'Rajasthan Royals', 'Delhi Capitals'
    ]
    matches = matches[matches['team1'].isin(eligible_teams) & matches['team2'].isin(eligible_teams)]
    
    # Extract First Innings Targets
    total_score_df = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
    first_innings_score = total_score_df[total_score_df['inning'] == 1].copy()
    first_innings_score['target'] = first_innings_score['total_runs'] + 1
    
    # Merge targets into match summary dataframe
    match_df = matches.merge(first_innings_score[['match_id', 'target']], left_on='id', right_on='match_id')
    
    # Filter deliveries to only include the 2nd innings (the run chase scenario)
    chase_deliveries = deliveries[deliveries['inning'] == 2].copy()
    
    # Pre-calculate metrics directly on deliveries dataset to avoid suffix naming errors
    chase_deliveries['current_score'] = chase_deliveries.groupby('match_id')['total_runs'].cumsum()
    chase_deliveries['player_dismissed'] = chase_deliveries['player_dismissed'].fillna("0").apply(lambda x: 0 if x == "0" else 1)
    chase_deliveries['wickets_fallen'] = chase_deliveries.groupby('match_id')['player_dismissed'].cumsum()
    
    # Now merge metrics with the match targets summary dataframe
    delivery_df = match_df.merge(chase_deliveries, left_on='id', right_on='match_id')
    
    # Calculate live operational metrics for ML features
    delivery_df['runs_left'] = delivery_df['target'] - delivery_df['current_score']
    delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])
    delivery_df['wickets_left'] = 10 - delivery_df['wickets_fallen']
    
    delivery_df['crr'] = (delivery_df['current_score'] * 6) / (120 - delivery_df['balls_left'])
    delivery_df['rrr'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']
    
    delivery_df['result'] = delivery_df.apply(lambda row: 1 if row['batting_team'] == row['winner'] else 0, axis=1)
    
    # Filter valid final features for modeling
    final_df = delivery_df[['batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets_left', 'target', 'crr', 'rrr', 'result']].dropna()
    final_df = final_df[final_df['balls_left'] != 0]
    
    X = final_df.drop(columns=['result'])
    y = final_df['result']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Encode Categories (Teams & Cities) into numbers for Logistic Regression
    trf = ColumnTransformer([('trf', OneHotEncoder(sparse_output=False, drop='first'), ['batting_team', 'bowling_team', 'city'])], remainder='passthrough')
    pipe = Pipeline(steps=[('step1', trf), ('step2', LogisticRegression(solver='liblinear'))])
    
    print("🧠 Training Logistic Regression model...")
    pipe.fit(X_train, y_train)
    return pipe

# Initialize the model pipeline globally
model_pipeline = None

# ==============================================================================
# FLASK BACKEND ROUTE API
# ==============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Safe float parsing to clean up incoming user strings (e.g. '12.0')
    overs_float = float(data['overs'])
    
    # Process user inputs dynamically into ML metric features
    balls_completed = (int(overs_float) * 6) + int((overs_float % 1) * 10)
    balls_left = 120 - balls_completed
    runs_left = int(data['target']) - int(data['score'])
    wickets_left = 10 - int(data['wickets'])
    crr = (int(data['score']) * 6) / balls_completed if balls_completed > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0
    
    # Multi-line dictionary structure prevents unterminated string errors
    input_df = pd.DataFrame({
        'batting_team': [data['batting_team']], 
        'bowling_team': [data['bowling_team']], 
        'city': [data['city']],
        'runs_left': [runs_left], 
        'balls_left': [balls_left], 
        'wickets_left': [wickets_left],
        'target': [data['target']], 
        'crr': [crr], 
        'rrr': [rrr]
    })
    
    # Extract structural probability breakdown
    probabilities = model_pipeline.predict_proba(input_df)
    loss_prob = round(probabilities[0][0] * 100, 2)
    win_prob = round(probabilities[0][1] * 100, 2)
    
    return jsonify({
        'batting_win': win_prob,
        'bowling_win': loss_prob
    })

if __name__ == '__main__':
    print("🏁 Script initialized...")
    model_pipeline = train_ipl_model()
    print("✅ ML Pipeline loaded successfully!")
    print("🌍 Starting local server on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)