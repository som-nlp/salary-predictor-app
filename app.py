import streamlit as st
import pandas as pd
import pickle

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data.csv')
    return data

def main():
    st.title('Salary Predictor')

    # Load data
    data = load_data()

    # Input for years of experience
    experience = st.slider('Years of Experience', min_value=0, max_value=20, value=5)

    # Load the model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predict salary
    salary_pred = model.predict([[experience]])

    # Display predicted salary
    st.write(f'Predicted Salary: Rs. {salary_pred[0]:,.2f}')

    st.markdown("<br><br>", unsafe_allow_html=True)


    st.text('Made with ❤ by Parth S. Aland')


if __name__ == '__main__':
    main()
