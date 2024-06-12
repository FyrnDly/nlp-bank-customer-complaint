import re
import pickle
import string
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow as tf
import tensorflow_hub as tf_hub
from tf_keras.models import Sequential, load_model
from tf_keras.utils import to_categorical
from tf_keras.layers import TextVectorization, Embedding, Dense, LSTM, Dropout, BatchNormalization
from tf_keras.callbacks import EarlyStopping, LearningRateScheduler, Callback
from tf_keras.metrics import Precision, Recall, AUC


# Data Example
@st.cache_data
def load_data():
    data = pd.read_csv('.\\models\\complaints.csv', index_col=0)
    data.columns = ['label','complaints']
    data.dropna(how="any", axis=0, inplace=True)
    data.drop_duplicates(subset='complaints', keep='first', inplace=True)
    
    def normalize_text(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub(r'\b(\w*x\w*){3,}\b', '', text, flags=re.IGNORECASE)

        return text
    data['complaints'] = data['complaints'].apply(normalize_text)
    data['label'] = data.label.map({'credit_card':'Credit Card','credit_reporting':'Credit Reporting','debt_collection':'Debt Collection','mortgages_and_loans':'Mortgages & Loans','retail_banking':'Retail Banking'})
    data['complaints_len'] = data['complaints'].apply(lambda x: len(x.split(' ')))
    return data

@st.cache_data
def get_predict(*input_text):
    # Model Prediction
    model = load_model('.\\models\\model_lstm_rus.h5')

    # Vectorizer Text
    from_disk = pickle.load(open('.\\models\\vectorizer.pkl', "rb"))
    vectorize = TextVectorization.from_config(from_disk['config'])
    vectorize.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    vectorize.set_weights(from_disk['weights'])
    
    text_vect = vectorize(input_text)
    score = model.predict(text_vect)[0]
    index = score.argmax()
    
    # Get result predict
    if index == 0:
        result = 'Credit Card'
    elif index == 1:
        result = 'Credit Reporting'
    elif index == 2:
        result = 'Debt Collection'
    elif index == 3:
        result = 'Mortgages & Loans'
    elif index == 4:
        result = 'Retail Banking'
        
    # Get percentage
    percentage = {
        'Credit Card' : f'{score[0]:.2f}%',
        'Credit Reporting' : f'{score[1]:.2f}%',
        'Debt Collection' : f'{score[2]:.2f}%',
        'Mortgages & Loans' : f'{score[3]:.2f}%',
        'Retail Banking' : f'{score[4]:.2f}%'
    }
    
    return result, percentage