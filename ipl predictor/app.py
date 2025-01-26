import streamlit as st
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define teams and cities
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Streamlit app title
st.title('IPL Win Predictor')

# Create input columns
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))
target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

if st.button('Predict Probability'):
    # Calculate additional inputs
    runs_left = target - score
    balls_left = 120 - int(overs * 6)
    remaining_wickets = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

    # Prepare the input DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [remaining_wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Ensure all inputs are of the correct type
    st.write("Input DataFrame:")
    st.write(input_df)

    try:
        # Inspect the pipeline steps and expected input
        st.write("Pipeline Steps:")
        for name, step in pipe.named_steps.items():
            st.write(f"Step name: {name}, Step type: {type(step)}")

        # Predict probabilities
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Display results
        st.header(f"{batting_team} - {round(win * 100, 2)}%")
        st.header(f"{bowling_team} - {round(loss * 100, 2)}%")
    except AttributeError as e:
        st.error(f"An attribute error occurred during prediction: {e}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
