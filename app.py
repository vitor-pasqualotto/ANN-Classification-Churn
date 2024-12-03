import tensorflow
import pandas
import numpy
import pickle
import streamlit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model # type: ignore

# Load the trained model
model = load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
streamlit.title('Customer Churn Prediction')

# User input
geography = streamlit.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = streamlit.selectbox('Gender', label_encoder_gender.classes_)
age = streamlit.slider('Age', 18, 92)
balance = streamlit.number_input('Balance')
credit_score = streamlit.number_input('Credit Score')
estimated_salary = streamlit.number_input('Estimated Salary')
tenure = streamlit.slider('Tenure', 0, 10)
num_of_products = streamlit.slider('Number of Products', 1, 4)
has_cr_card = streamlit.selectbox('Has Credit Card', [0, 1])
is_active_member = streamlit.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pandas.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    
})

# One Hot encode'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]])
geo_encoded_df = pandas.DataFrame(geo_encoded.toarray(), columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one hot encoded columns with input data 
input_data = pandas.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

streamlit.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    streamlit.write('The customer is likely to churn.')
else:
    streamlit.write('The customer is not likely to churn.')