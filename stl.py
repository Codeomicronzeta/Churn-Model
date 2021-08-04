import numpy as np
import pandas as pd
import streamlit as st
from rfmodel import model_pi


def main():
    st.title('Churn Predictor')
    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "color:white;text-align:center;">Streamlit CHurn Predictor Application</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    age =  st.text_input('Age', 'Enter Age')
    nop = st.text_input('Number of Products', 'Enter the Number of Products')
    estsal = st.text_input('Estimated Salary', 'Enter the Estimated Salary')
    credscore = st.text_input('Credit Score', 'Enter the Credit Score')
    bal = st.text_input('Balance', 'Enter the Balance')
    tenure = st.text_input('Tenure', 'Enter the Tenure')
    result = ""
    input_data = np.array([[age, nop, estsal, credscore, bal, tenure]])
    if st.button('Predict Churn'):
        result = model_pi.predict(input_data)
        if(result == [0]):
            result = 'Not Churned'
        else:
            result = 'Churned'
    st.success(f"The output is {result}")

if __name__=='__main__':
    main()

