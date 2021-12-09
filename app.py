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
from car_predict.car_predict_app import run_CAR_PERDICT_app
def main():
    
    # 사이드바 메뉴
    menu = ['Home','자동차 구매 예측','ML']
    choice = st.sidebar.selectbox('Menu',menu)

    if choice == 'Home' :
        st.title("Minhyun's Project")
        st.markdown( '## hello')

    elif choice == '자동차 구매 예측' :
        run_CAR_PERDICT_app()






if __name__=='__main__':
    main()