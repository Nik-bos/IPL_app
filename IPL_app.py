import pandas as pd
import streamlit as st
import pickle
import joblib
import os

teams = ['Gujarat Titans', 'Rajasthan Royals', 'Punjab Kings',
       'Mumbai Indians', 'Royal Challengers Bangalore',
       'Sunrisers Hyderabad', 'Delhi Capitals', 'Chennai Super Kings',
       'Kolkata Knight Riders']

cities = ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Pune', 'Dubai',
       'Sharjah', 'Abu Dhabi', 'Delhi', 'Chennai', 'Hyderabad',
       'Visakhapatnam', 'Chandigarh', 'Bengaluru', 'Jaipur', 'Indore',
       'Bangalore', 'Kanpur', 'Rajkot', 'Raipur', 'Ranchi', 'Cuttack',
       'Dharamsala', 'Nagpur', 'Johannesburg', 'Centurion', 'Durban',
       'Bloemfontein', 'Port Elizabeth', 'Kimberley', 'East London',
       'Cape Town', 'Guwahati']

# model_path = 'IPL_prediction_using_StackingClf.pkl'
# pipe = pickle.load(open(model_path, 'rb'))
pipe = joblib.load(r'ipl_predictor.joblib')

# Starting to build app

st.title("IPL Win Predictor")

col1, col2 = st.columns(2)    # Instead of beta_columns() new version has columns()

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

col_3, col_4, col_5 = st.columns(3)

with col_3:
    city = st.selectbox("Host city", sorted(cities))
with col_4:
    runs_left = st.number_input("Runs left")
with col_5:
    balls_left = st.number_input('Balls left')
    

col_6, col_7 = st.columns(2)

with col_6:
    wickets_remaining = st.number_input("Wickets left")
with col_7:
    total_run_x = st.number_input('Target')


if st.button('Predict'):
    balls_played = 120 - balls_left
    runs_made = total_run_x - runs_left
    # Avoid division by zero errors
    crr = runs_made / (balls_played / 6) if balls_played > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else float('inf')  # Set rrr to 'inf' if balls_left is 0

    # Creating dataframe

    data = {
    'batting_team': [batting_team],
    'bowling_team': [bowling_team],
    'city': [city],
    'runs_left': [runs_left],
    'balls_left': [balls_left],
    'wickets_remaining': [wickets_remaining],
    'total_run_x': [total_run_x],
    'crr': [crr],
    'rrr': [rrr]
}

    df = pd.DataFrame(data)

    st.table(df)    # Checking if dataframe is created properly

    pred = pipe.predict(df)
    if pred == 1:
        st.header(batting_team + 'will win.')
    else:
        st.header(bowling_team + ' will win.')

# Having some problem after deploying. Render deployed and provided the link  as well
# but due to some problem of port, it was not working. So, entered below code from ChatGPT>
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'

