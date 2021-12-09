import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
import pickle
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

def run_CAR_PERDICT_app() :
    st.title("자동차 구매 예측 딥러닝") 
    st.write('고객데이터와 자동차 구매 데이터에 대한 내용입니다. 해당 고객의 정보를 입력하면, 얼마정도의 차를 구매 할 수 있는지를 예측합니다.')
    selected = st.selectbox('자동차 구매 예측',['EDA','Predict'])
    
    if selected == 'EDA' : 
        st.subheader('학습자료 EDA 입니다.')

        car_df = pd.read_csv('car_predict/data/Car_Purchasing_Data.csv', encoding='ISO-8859-1')
    
        radio_menu = ['데이터프레임','통계치']
        selected_radio = st.radio('',radio_menu)

        if selected_radio == '데이터프레임' :
            st.dataframe(car_df)

        elif selected_radio == '통계치' :
            st.dataframe(car_df.describe())

    elif selected == 'Predict' :
        st.subheader('Machine Learning')
        Gender = st.radio('성별을 입력하세요',('남성','여성'),index=1)
        if Gender == '남성': Gender = 1
        else : Gender = 0
        Age = st.number_input('나이를 입력하세요',0,100,value = 38)
        Annual_Salary =	st.number_input('연봉을 달러로 환산하여 입력하세요',value = 90000)
        Card_Debt =	st.number_input('카드빛을 달러로 환산하여 입력하세요', value = 2000)
        Net_Worth =st.number_input('순자산을 달러로 환산하여 입력하세요', value= 500000)
        user = [Gender,  Age, Annual_Salary,Card_Debt,Net_Worth]
        
        print(user)

        Pred= st.button('Predict')
        
        if Pred :
            print(user)
            with st.spinner('Wait for it...'):
                model = tensorflow.keras.models.load_model('car_predict/data/car_ai.h5')

                new_data  = np.array(user)

                new_data = new_data.reshape(1,-1)

                sc_X = joblib.load('car_predict/data/sc_X.pkl')

                new_data = sc_X.transform(new_data)

                y_pred = model.predict(new_data)

                sc_y = joblib.load('car_predict/data/sc_y.pkl')

                y_pred_original = sc_y.inverse_transform(y_pred)

            st.success('고객이 선호하는 구매가는 '+str(y_pred_original[0,0])+' 입니다.')

    