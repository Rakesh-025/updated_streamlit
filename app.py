
import numpy as np
import pickle
import pandas as pd
#from flasgger import Swagger
import streamlit as st 
#pip install streamlit
from PIL import Image


#app=Flask(__name__)
#Swagger(app)

pickle_in = open("model_Prediction.pkl","rb")
classifier=pickle.load(pickle_in)

#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_note_authentication(Maintenance_cost,Marketing_cost,Debentures,Duration_of_coaching_in_Hours,Profit_Margin):
    
    """Let's Predict the Price of New Product 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: Maintenance_cost
        in: query
        type: number
        required: true
      - name: Marketing_cost
        in: query
        type: number
        required: true
      - name: Debentures
        in: query
        type: number
        required: true
      - name: Duration_of_coaching_in_Hours
        in: query
        type: number
        required: true
      - name: Profit_Margin
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[Maintenance_cost,Marketing_cost,Debentures,Duration_of_coaching_in_Hours,Profit_Margin]])
    
    print(prediction)
    return prediction



def main():
    st.title("Price Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Price Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Maintenance_cost = st.text_input("Maintenance_cost","Type Here")
    Marketing_cost = st.text_input("Marketing_cost","Type Here")
    Debentures = st.text_input("Debentures","Type Here")
    Duration_of_coaching_in_Hours = st.text_input("Duration_of_coaching_in_Hours","Type Here")
    Profit_Margin = st.text_input("Profit_Margin","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Maintenance_cost,Marketing_cost,Debentures,Duration_of_coaching_in_Hours,Profit_Margin)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
    
    
