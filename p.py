import streamlit as st
import pandas as pd
import pickle
import numpy as np

st.write("Zomato app")
df = pd.read_csv(r"D:\car insurance claim\zomato_df_1.csv")

# Encoding function
def encode_input(value, unique_values):
    # Create a mapping from the original value to the encoded integer
    value_map = {val: idx for idx, val in enumerate(unique_values)}
    return value_map.get(value, -1)  # Return -1 if the value is not found

def predict_rating(online, book_table, votes, location, rest_type, cuisine, cost, type):
    with open(r"D:\car insurance claim\model.pkl", "rb") as f:
        random_forest = pickle.load(f)
    
    # Ensure all inputs are encoded as they were when the model was trained
    user_data = np.array([[online, book_table, votes, location, rest_type, cuisine, cost, type]])
    
    y_predict = random_forest.predict(user_data)
    
    return y_predict

# Get user inputs and encode them
online = encode_input(st.selectbox("Online Order", df["online_order"].unique()), df["online_order"].unique())
book_table = encode_input(st.selectbox("Book Table", df["book_table"].unique()), df["book_table"].unique())
votes = st.number_input("Votes", min_value=int(df["votes"].min()), max_value=int(df["votes"].max()), value=int(df["votes"].median()))
location = encode_input(st.selectbox("Location", df["location"].unique()), df["location"].unique())
rest_type = encode_input(st.selectbox("Restaurant Type", df["rest_type"].unique()), df["rest_type"].unique())
cuisine = encode_input(st.selectbox("Cuisines", df["cuisines"].unique()), df["cuisines"].unique())
cost = st.number_input("Cost", min_value=int(df["cost"].min()), max_value=int(df["cost"].max()), value=int(df["cost"].median()))
type = encode_input(st.selectbox("Type", df["type"].unique()), df["type"].unique())

button1 = st.button("Encode To Numeric")
if button1:
    def encode(df):
        for column in df.columns[~df.columns.isin(["rate", "cost", "votes"])]:
            df[column] = df[column].factorize()[0]
        return df

    zomato_en = encode(df.copy())
    
button2 = st.button("Predict")
if button2:
    rate = predict_rating(online, book_table, votes, location, rest_type, cuisine, cost, type)
    rounded_rate = round(rate[0], 2)
    
    # Display the rounded rating in big green text
    st.markdown(f"<h2 style='color: green;'>**Rating is: {rounded_rate}**</h2>", unsafe_allow_html=True)
