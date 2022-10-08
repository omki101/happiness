import numpy as np
import streamlit as st
import pickle




loaded_model=pickle.load(open('C:/Users/Omkar/OneDrive/Documents/Happiness INdex/trained model.sav','rb'))

#creating a function for prediction
def happiness_index(input_data):
    input_data=(12,0.9,70,0.9)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if(prediction<5):
        print("You are not that happy")
    else:
        print("You are the quite happy")

    
def main():
    st.title('Happiness Index')
    
    #getting the input data from user
    
    
    GDPPerCapita = st.text_input('GDP Per Capita')
    SocialSupport = st.text_input('Social Support')
    HealthyLifeExpectancy=st.text_input('Healthy Life Expectancy')
    FreedomtomakeLifeChoices=st.text_input(' Freedom to make Life Choices')
    
    
      #code for prediction
    HappinessIndex = ''
      
      
      #creating a button for Prediction
    if st.button('Result'):
               HappinessIndex = happiness_index([GDPPerCapita,SocialSupport,HealthyLifeExpectancy,FreedomtomakeLifeChoices]) 
          
    st.success(HappinessIndex)
      
      
if __name__ == '__main__':
    main()
    