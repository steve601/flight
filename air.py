import streamlit as st 
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

transformer =ColumnTransformer(transformers=[('cat',OrdinalEncoder(),[0,1,2,3,4,5,6])],remainder = 'passthrough')
model2 = joblib.load(open('C:/Users/odhia/OneDrive/Desktop/streamlit tut/air_model.pkl','rb'))
st.title('FLIGHT PRICE PREDICTION APP')
def main():
    st.write('Fill the values in order to predict the price')
    airline = st.selectbox('Type Of Airline',('SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo','Air_India'))
    source_city = st.selectbox('Flying from...',('Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai'))
    departure_time = st.selectbox('Departure Time',('Evening', 'Early_Morning', 'Morning', 'Afternoon', 'Night','Late_Night'))
    stops = st.selectbox('Stops',('zero','one','two'))
    arrival_time = st.selectbox('Arrival Time',('Night', 'Morning', 'Early_Morning', 'Afternoon', 'Evening','Late_Night'))
    destination_city = st.selectbox('Flying to...',('Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai', 'Delhi'))
    Class = st.selectbox('Class',('Economy', 'Business'))
    days_left = st.number_input('How many days are left to the date of travel?',1,49)
    
    if st.button('PREDICT'):
        p = np.array([[airline,source_city,departure_time,stops,arrival_time,destination_city,Class,days_left]])
        p = transformer.fit_transform(p)
        p = p.astype(float)
        
        fare = model2.predict(p)
        st.subheader(f'The estimated flight price is: ${fare[0]:.2f}')
        
        
if __name__ == '__main__':
    main()