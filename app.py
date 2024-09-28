import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('model.joblib')

# Streamlit UI
st.set_page_config(page_title="Palmer Penguin Prediction", layout="wide")
st.title("🐧 Palmer Penguin Prediction")
st.header("Enter Penguin Features")


# Sliders for the user input
culmen_length = st.slider("Culmen Length (mm)", min_value=30, max_value=60, value=45)
culmen_depth = st.slider("Culmen Depth (mm)", min_value=10, max_value=25, value=17)
flipper_length = st.slider("Flipper Length (mm)", min_value=170, max_value=240, value=210)
body_mass = st.slider("Body Mass (g)", min_value=2600, max_value=6400, value=3500)

# Button for prediction
if st.button("Predict", key='predict_button'):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Culmen Length (mm)': [culmen_length],
        'Culmen Depth (mm)': [culmen_depth],
        'Flipper Length (mm)': [flipper_length],
        'Body Mass (g)': [body_mass]
    })
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Mapping the predicted class to the Palmer Penguin names
    penguin_species = ['Adelie Penguin', 'Chinstrap Penguin', 'Gentoo Penguin']
    predicted_species = penguin_species[prediction[0]]

    penguin_images = {
        'Adelie Penguin': 'images/Adelie Penguin.jpeg',
        'Chinstrap Penguin': 'images/Chinstrap penguin.jpeg',
        'Gentoo Penguin': 'images/Gentoo penguin.jpeg'
    }

    # Display results
    
    col1,col2,col3 = st.columns([1,2,1])
    col2.info(f'The Predicted Penguin Species is: **{predicted_species}**')
    col2.image(penguin_images[predicted_species], caption=predicted_species)

# Footer
st.markdown("---")
st.write("Explore the penguin species and learn more!")
