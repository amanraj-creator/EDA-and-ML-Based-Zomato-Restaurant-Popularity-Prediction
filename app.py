import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model and scaler
rf_model = joblib.load("zomato_popularity_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")  # Label encoder for 'listed_in(type)'

#Loading image
from PIL import Image
# Open and resize the image
image = Image.open("Logo.jpg")
resized_image = image.resize((200, 140))  # Resize 

# Center image using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(resized_image, caption="WELCOME TO RESTAURANT POPULARITY PREDICTOR APP!🥂", use_container_width=False)



# Streamlit UI Inputs
st.title("  🍽️Zomato's Restaurant Popularity Predictor🍕 ")

online_order = st.selectbox("Does the restaurant accept Online Orders?", ["Yes", "No"])
book_table = st.selectbox("Does the restaurant allow Table Booking?", ["Yes", "No"])
rate = st.slider("Restaurant Rating (Out of 5)", 1.0, 5.0, 4.0)
approx_cost = st.number_input("Approximate Cost for Two People", min_value=100, max_value=5000, value=500)

# Categorical feature
listed_in_type = st.selectbox("Restaurant Type", ["Casual Dining", "Cafe", "Fine Dining", "Quick Bites", "Dessert Parlour"])  # Update as per dataset

# Encode categorical feature
if listed_in_type in le.classes_:
    listed_in_type = le.transform([listed_in_type])[0]
else:
    listed_in_type = 0  # Default encoding if not found

# Compute additional features
popularity_score = rate * approx_cost

# Cost and Rating Category (Example: Define logic based on training)
cost_category = 1 if approx_cost < 500 else (2 if approx_cost < 1500 else 3)
rating_category = 1 if rate < 3 else (2 if rate < 4 else 3)

# Convert categorical to numeric
online_order = 1 if online_order == "Yes" else 0
book_table = 1 if book_table == "Yes" else 0

# Prepare input data (Matches 8 training features)
input_data = np.array([[online_order, book_table, rate, approx_cost, 
                        listed_in_type, popularity_score, cost_category, rating_category]])

 # Convert input_data into a DataFrame before scaling
input_data = pd.DataFrame(input_data, columns=['online_order', 'book_table', 'rate', 'approx_cost(for two people)',
                                               'listed_in(type)', 'popularity_score', 'cost_category', 'rating_category'])

# Now scale the DataFrame (instead of NumPy array)
input_data = scaler.transform(input_data)


# Predict
if st.button("Predict Popularity"):
    prediction = rf_model.predict(input_data)
    st.success(f"📊 Predicted Votes (Popularity Score): {int(prediction[0])}")
