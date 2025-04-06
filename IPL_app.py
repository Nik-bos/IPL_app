import pandas as pd
import streamlit as st
import pickle
import joblib
import time

'''
We don't need time to include it in req.txt bcz time is a python
built-in package not a third party module/package.
time, os, sys, math etc are python's built-in modules/packages
'''
# Create a placeholder
placeholder = st.empty()

# Show waking-up message in the placeholder
with placeholder.container():
    st.spinner("Waking up the app...")  # Spinner context must be inside the block too
    st.title("Hey there! 👋")
    st.subheader("Wish you a very happy and a beautiful day!")
    st.write("The app is waking up — please hang tight for a few seconds...")
    time.sleep(5)  # Simulate loading time

# Now clear the placeholder (removes the loading message)
placeholder.empty()

# Above code lines will print welcome msg till the app loads.
# If You dont know this codes, it's ok, not needed compulsorily.


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

    # st.table(df)    # Checking if dataframe is created properly

    pred = pipe.predict(df)
    if pred == 1:
        st.header(batting_team + 'will win.')
    else:
        st.header(bowling_team + ' will win.')


