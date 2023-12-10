#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


#load the model from disk
import joblib
model = joblib.load(r"./notebook/model.sav")

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Customer Insight Engine')

      #Setting Application description
    st.markdown("""
      Our customer churn prediction model empowers telecom companies with advanced analytics to proactively identify and retain customers at risk of churning. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('./Images/churn.jpeg')
    add_selectbox = st.sidebar.selectbox(
	"How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('Customer Churn prediction tool for telecom companies')
    st.sidebar.image(image)
    

    if add_selectbox == "Online":
        #st.info("Input data below")
       
        
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))
        partners = st.selectbox('Partner:', ('Yes', 'No'))


        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
        
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
        deviceprotection=st.selectbox("Does the customer have device protection", ('Yes','No','No internet service'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
        PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
       

        data = {
                'SeniorCitizen': seniorcitizen,
                'Partner':partners,
                'Dependents': dependents,
                'tenure':tenure,
                'PaperlessBilling': paperlessbilling,
                'MonthlyCharges': monthlycharges, 
                'TotalCharges': totalcharges,
                'Contract': contract,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'DeviceProtection':deviceprotection,
                'TechSupport': techsupport,
                'PaymentMethod':PaymentMethod, 
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below ')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service :worried:')
            else:
                st.success('No, the customer is happy with your Services :smiley:')
        

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
       
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
             
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.', 
                                                    0:'No, the customer is happy with Telco Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()




