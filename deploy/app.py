import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from tensorflow .keras.models import load_model

import warnings
warnings.filterwarnings(action='ignore')

st.header('Milestone 1 Phase 2')
st.write("Nama: Risqi Wahyu Permana")
st.write("Batch: HCK 006")

#load data
@st.cache_data
def fetch_data():
    df = pd.read_csv('https://raw.githubusercontent.com/wahyupermana-10/dataset/main/churn.csv')
    return df

df = fetch_data()

page = st.sidebar.selectbox('Choose a page', ['EDA', 'Prediction'])

if page == 'EDA':
    st.title('Exploratory Data Analysis')

    st.subheader('Age Distribution')
    #see age distribution use barplot
    plt.figure(figsize=(15,5))
    sns.barplot(x=df['age'].value_counts().index, y=df['age'].value_counts().values)
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('Age Distribution')
    st.pyplot(plt)
    st.write('It turns out that the age of product users is in the age range of 10 years to 64 years and has a uniform distribution.')
    st.write('')

    st.subheader("Member Category Distribution")
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(121)
    ax.pie(df['membership_category'].value_counts().values, labels=df['membership_category'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette('Set3'))
    ax.set_title('Member Category Distribution')
    st.pyplot(plt)
    st.write('The top 3 membership categories are Basic Membership, No Membership, and Gold Membership.')
    st.write('')

    st.subheader("Member Category Distribution")
    #see Which category of members quit the most?
    plt.figure(figsize=(15,5))
    sns.countplot(x=df['membership_category'], hue=df['churn_risk_score'])
    plt.xlabel('Member Category')
    plt.ylabel('Count')
    plt.title('Member Category Distribution')
    st.pyplot(plt)
    st.write('The customers who leave the most are those who do not have a membership and those who have a basic membership.')
    st.write('')

    st.subheader("Complaint Status Distribution")
    #What are the complaints and amounts given by customers?
    plt.figure(figsize=(15,5))
    sns.countplot(x=df['complaint_status'], hue=df['churn_risk_score'])
    plt.xlabel('Complaint Status')
    plt.ylabel('Count')
    plt.title('Complaint Status Distribution')
    st.pyplot(plt)
    st.write('From here the above plot can be seen, the customers who leave and stay are actually balanced. However, we get information that there are still many complaints that are not applicable.')
    st.write('')

    st.subheader("Used Special Discount Distribution")
    plt.figure(figsize=(15,5))
    sns.countplot(x=df['used_special_discount'], hue=df['churn_risk_score'])
    plt.xlabel('Used Special Discount')
    plt.ylabel('Count')
    plt.title('Used Special Discount Distribution')
    st.pyplot(plt)
    st.write('Basic membership, no membership, gold membership are the top 3 membership categories.')
else:
    st.title('Prediction')
    st.write('Please input your data')
    feedback = st.selectbox('feedback', df['feedback'].unique())
    points_in_wallet = st.number_input('points_in_wallet', value=0)
    avg_transaction_value = st.number_input('avg_transaction_value', value=0)
    membership_category = st.selectbox('membership_category', df['membership_category'].unique())
    avg_frequency_login_days = st.number_input('avg_frequency_login_days', value=0)
    data = {
        'feedback' : feedback,
        'points_in_wallet' : points_in_wallet,
        'avg_transaction_value' : avg_transaction_value,
        'membership_category' : membership_category,
        'avg_frequency_login_days' : avg_frequency_login_days
    }
    input = pd.DataFrame(data, index=[0])

    st.subheader('Predict')
    st.write(input)

    load_model = load_model("ann_wq5.h5")
    transform = joblib.load('preprocess.pkl')

    if st.button('Predict'):
        change = transform.transform(input)
        pred = load_model.predict(change)
        results = np.where(pred >= 0.5, 1, 0)

        if results == 1:
            results = 'Churn'
        else:
            results = 'Does not churn'

        st.write('Based on the input, the placement model predicted: ')
        st.write(results)