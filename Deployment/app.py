from flask import Flask, render_template, request, redirect, jsonify
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn import linear_model
#from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
import io
import csv
import os
import pandas as pd
import time
import numpy as np
import zipfile
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import re
from datetime import datetime, date
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from scipy.sparse import hstack
from scipy.sparse import vstack
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from scipy.stats import randint as sp_randint
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pickle
import joblib
from IPython.display import Image



import flask
app = Flask(__name__)


def dcg_score(y_true, y_score, k=5):
    
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    
    """Normalized discounted cumulative gain (NDCG) at rank K.

    Normalized Discounted Cumulative Gain (NDCG) measures the performance of a
    recommendation system based on the graded relevance of the recommended
    entities. It varies from 0.0 to 1.0, with 1.0 representing the ideal
    ranking of the entities.

    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities.
    k : int
        Rank.

    Returns
    -------
    score : float

    Example
    -------
    >>> ground_truth = [1, 0, 2]
    >>> predictions = [[0.15, 0.55, 0.2], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    1.0
    >>> predictions = [[0.9, 0.5, 0.8], [0.7, 0.2, 0.1], [0.06, 0.04, 0.9]]
    >>> score = ndcg_score(ground_truth, predictions, k=2)
    0.6666666666
    """
    
    lb = LabelBinarizer()
    lb.fit(range(predictions.shape[1] + 1))
    T = lb.transform(ground_truth)

    scores = []

    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


# NDCG Scorer function
ndcg_scorer = make_scorer(ndcg_score, needs_proba=True, k=5)

def func_age_imput_median(age):
    
    """
    Function to replace any value of age less than 15 or greater than 2007 with the median age
    
    parameters: age 
    
    returns : age  
    
    """
    
    if age < 15.0 or age > 2007.0:
        return 34.0
    else:
        return age


def func_age_imput_year(age,year):
    
    """
    Function to check if age is between 117 and 2007. If true, it will substract that age value from the year of account 
    creation to get the exact age of the user on the year he created the account.
        
    parameters: age,account_created_year 
    
    returns : age  
    
    """
    
    res = 0
    if age > 117.0 and age <=2007.0:
        res = year-age
        return res 
    else:
        return age

def total_sec(sec_list):
    
    """
    Function to calculate total time a particular user has spent in accessing the application.
        
    parameters: secs_elapsed 
    
    returns : total_secs_elapsed  
    
    """
    
    res = []
    
    #print(sec_list)
    
    
    #need to convert each element of the list to string so as to replace the 'nan' with '0' so that we dont get ValueError later.
    sec_list = [ str(i) for i in sec_list ] 
    sec_list = [ re.sub('nan','0',i) for i in sec_list] 
    
    #sec_list is a list of strings now. Iterating over the elements of the list.
    for i in sec_list:
        
        #print(i)
        #Converting the string to float to take the sum later
        res.append(float(i)) 
    res = sum(res)
    return res

from statistics import mean

def average_sec(sec_list):
    
    """
    Function to calculate average time a particular user has spent in accessing the application.
        
    parameters: secs_elapsed 
    
    returns : mean_secs_elapsed  
    
    """
    
    res = []
    
    #need to convert each element of the list to string so as to replace the 'nan' with '0' so that we dont get ValueError later.
    sec_list = [ str(i) for i in sec_list ] 
    sec_list = [ re.sub('nan','0',i) for i in sec_list] 
    
    #session_df is a list of strings now. Iterating over the elements of the list.
    for i in sec_list:
        
        #Converting the string to float to take the mean later
        res.append(float(i)) 
    res = mean(res)
    return res

def session_count(sec_list):
    
    """
    Function to calculate total sessions a particular user has used in accessing the application.
        
    parameters: secs_elapsed 
    
    returns : session_count  
    
    """
        
    return len(sec_list)

def unique_action(action):
    
    """
    Function to calculate unique actions per user.
        
    parameters: action/device_* 
    
    returns : unique_action/device_*  
    
    """
   
    action = [str(i) for i in action]
    
    action = [re.sub('nan','na',i) for i in action]
    
    action = ','.join(set(action))
    
    return action

def preprocess_data(data):
    
    """
    Function to preprocess the raw data.
    
    It performs the following tasks :
    
    a) Removing outliers
    b) Filling Missing Values
    c) Finding unique values in the sessions columns
    d) Computing sum/mean for the secs_elapsed column
    e) Dropping the columns not needed
    
    parameters: raw data
    
    returns : processed data  
    
    """
    
    #Processing 'date_account_created' :
    data['date_account_created'] = pd.to_datetime(data['date_account_created'])
    data['account_created_day'] = data['date_account_created'].dt.weekday
    data['account_created_month'] = data['date_account_created'].dt.month
    data['account_created_year'] = data['date_account_created'].dt.year
    
    print('"date_account_created" field is finished processing...')
    
    #Processing 'age' :
    data['age'] = data['age'].fillna(34.0)
    data['age'] = data['age'].apply(func_age_imput_median)
    data['age'] = data.apply(lambda x: func_age_imput_year(x['age'], x['account_created_year']), axis=1)
    
    print('"age" field is finished processing...')
    
    data['first_affiliate_tracked'] = data['first_affiliate_tracked'].fillna('untracked')
    
    print('"first_affiliate_tracked" field is finished processing...')
    
    data['Total_secs_elapsed'] = data['secs_elapsed'].apply(total_sec)
    data['Mean_secs_elapsed'] = data['secs_elapsed'].apply(average_sec)
    data['session_count'] = data['secs_elapsed'].apply(session_count)
    data['unique_action'] = data['action'].apply(unique_action)
    data['unique_action_type'] = data['action_type'].apply(unique_action)
    data['unique_action_detail'] = data['action_detail'].apply(unique_action)
    data['unique_device_type'] = data['device_type'].apply(unique_action)
    
    print('session fields are finished processing...')
    
    data = data.drop(['date_first_booking','timestamp_first_active','id','user_id','action','action_type',
                      'action_detail','device_type','secs_elapsed','date_account_created'],axis = 1, inplace = True)
    
    print('Fields already processed are dropped...')
    
    return data

def vectorise_data(data):
    
    """
    Function to vectorise the processed data and 
    stacking the sparse columns together to create the final data matrix.
    
    parameters: processed data
    
    returns :  vectorised data  
    
    """
    vect_gender = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_gender.pickle', 'rb'))
    vect_signup_method = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_signup_method.pickle', 'rb'))
    vect_language = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_language.pickle', 'rb'))

    vect_affiliate_channel = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_affiliate_channel.pickle', 'rb'))

    vect_affiliate_provider = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_affiliate_provider.pickle', 'rb'))
    vect_first_affiliate_tracked = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_first_affiliate_tracked.pickle', 'rb'))

    vect_signup_app = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_signup_app.pickle', 'rb'))
    vect_first_device_type = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_first_device_type.pickle', 'rb'))
    vect_first_browser = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_first_browser.pickle', 'rb'))

    vect_unique_device_type = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_unique_device_type.pickle', 'rb'))

    tfidf_vect_action = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\tfidf_vect_action.pickle', 'rb'))
    tfidf_vect_action_type = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\tfidf_vect_action_type.pickle', 'rb'))
    tfidf_vect_action_detail = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\tfidf_vect_action_detail.pickle', 'rb'))
    
    test_session_gender_bow = vect_gender.transform(data['gender'].values)
    print('Gender column vectorisation finished...')
    test_session_signup_method_bow = vect_signup_method.transform(data['signup_method'].values)
    print('signup_method column vectorisation finished...')
    test_session_language_bow = vect_language.transform(data['language'].values)
    print('language column vectorisation finished...')
    test_session_affiliate_channel_bow = vect_affiliate_channel.transform(data['affiliate_channel'].values)
    print('affiliate_channel column vectorisation finished...')
    test_session_affiliate_provider_bow = vect_affiliate_provider.transform(data['affiliate_provider'].values)
    print('affiliate_provider column vectorisation finished...')
    test_session_first_affiliate_tracked_bow = vect_first_affiliate_tracked.transform(data['first_affiliate_tracked'].values)
    print('first_affiliate_tracked column vectorisation finished...')
    test_session_signup_app_bow = vect_signup_app.transform(data['signup_app'].values)
    print('signup_app column vectorisation finished...')
    test_session_first_device_type_bow = vect_first_device_type.transform(data['signup_app'].values)
    print('signup_app column vectorisation finished...')
    test_session_first_browser_bow = vect_first_browser.transform(data['first_browser'].values)
    print('first_browser column vectorisation finished...')
    test_session_unique_device_type_bow = vect_unique_device_type.transform(data['unique_device_type'].values)
    print('unique_device_type column vectorisation finished...')
    test_session_action_tfidf = tfidf_vect_action.transform(data['unique_action'].values)
    print('unique_action column vectorisation finished...')
    test_session_action_type_tfidf = tfidf_vect_action_type.transform(data['unique_action_type'].values)
    print('unique_action_type column vectorisation finished...')
    test_session_action_detail_tfidf = tfidf_vect_action_detail.transform(data['unique_action_detail'].values)
    print('unique_action_detail column vectorisation finished...')
    
    print('Current Columns : ')
    print(data.columns)
    data = data.drop(['Unnamed: 0_y','Unnamed: 0_x','gender', 'country_destination','signup_method', 'language', 'affiliate_channel','affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser','unique_action','unique_action_type','unique_action_detail','unique_device_type'],axis=1)
    print('Dropping columns that are vectorised : ')
    print(data.columns)
    print(data.shape)
    
    # Stacking the numerical columns with the vectorised columns to create the final data matrix :
    data = sparse.hstack((data,test_session_gender_bow,test_session_signup_method_bow,test_session_language_bow,test_session_affiliate_channel_bow,test_session_affiliate_provider_bow,test_session_first_affiliate_tracked_bow,test_session_signup_app_bow,test_session_first_device_type_bow,test_session_first_browser_bow,test_session_unique_device_type_bow,test_session_action_tfidf,test_session_action_type_tfidf,test_session_action_detail_tfidf)).tocsr()
    print('Vectorisation finished...')
    print('Data ready for modelling....')
    
    return data

def final_method(data):
    
    """
    
    Function takes in raw data as input.
    and returns predictions for the input. 
    Here the input can be a single point or a set of points.
    
    parameters: raw data
    
    returns :  prediction for the data 
        
    """
    
    country_list = []
    start = time.process_time()
    print('START Preprocessing....\n')
    preprocess_data(data)
    print('\n START Vectorisation...\n')
    data1 = vectorise_data(data)
    
    print('\n START Modelling...\n')
    clf_rf = joblib.load(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\models\clf_rf')
    Y = pd.read_csv(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\test_users.csv\raw_target.csv')
    rf_pred = clf_rf.predict_proba(data1)
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    
    for i in rf_pred:
        
        country_list.append(le.inverse_transform(np.argsort(i)[::-1][:5]).tolist())
        
    print('\n PREDICTIONS....\n')
    print(country_list,rf_pred)
    print('Time Taken (in secs) to return the prediction : ',time.process_time() - start)
    
    return country_list,rf_pred

def final_method2(raw_label,pred):
    
    """
    
    Function takes raw label and corresponding prediction as input and returns ndcg score.
    
    parameters:  ground truth , prediction
    
    return : ndcg score 
        
    """
    print('NDCG SCORE : \n')
    score = ndcg_score(raw_label,pred,5)
    print(score)
    return score


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.errorhandler(Exception)
def server_error(err):
    return render_template('error.html',result = 'ERROR')

@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    uploaded_file_train = request.files['fileToUpload']
    uploaded_file_session = request.files['fileToUpload_session']
    
    if uploaded_file_train.filename != '':
        
        file_path = os.path.join(uploaded_file_train.filename)
        uploaded_file_train.save(file_path)

        file_path_session = os.path.join(uploaded_file_session.filename)
        uploaded_file_session.save(file_path_session)

        train_df = pd.read_csv(file_path)
        session_df = pd.read_csv(file_path_session)
        print(train_df.shape)
        print(session_df.shape)

    vect_gender = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_gender.pickle', 'rb'))
    vect_signup_method = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_signup_method.pickle', 'rb'))
    vect_language = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_language.pickle', 'rb'))

    vect_affiliate_channel = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_affiliate_channel.pickle', 'rb'))

    vect_affiliate_provider = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_affiliate_provider.pickle', 'rb'))
    vect_first_affiliate_tracked = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_first_affiliate_tracked.pickle', 'rb'))

    vect_signup_app = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_signup_app.pickle', 'rb'))
    vect_first_device_type = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_first_device_type.pickle', 'rb'))
    vect_first_browser = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_first_browser.pickle', 'rb'))

    vect_unique_device_type = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\vect_unique_device_type.pickle', 'rb'))

    tfidf_vect_action = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\tfidf_vect_action.pickle', 'rb'))
    tfidf_vect_action_type = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\tfidf_vect_action_type.pickle', 'rb'))
    tfidf_vect_action_detail = pickle.load(open(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\final_dataset\tfidf_vect_action_detail.pickle', 'rb'))
    Y = pd.read_csv(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\test_users.csv\raw_target.csv')

    #train_df = pd.read_csv(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\pickles\train_raw3.csv')
    #session_df = pd.read_csv(r'C:\Users\NamrataT\Desktop\CS_1\Dataset\airbnb-recruiting-new-user-bookings\pickles\session_raw3.csv')
    session_df_unq_rec1 = session_df.groupby('user_id', as_index=False).agg(lambda x: x.tolist())
    train_session_df = train_df.merge(session_df_unq_rec1, left_on = 'id', right_on = 'user_id', how = 'inner')

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y_raw = train_session_df['country_destination']
    Y_raw = le.transform(Y_raw)
    X_raw = train_session_df[:1]
    #try:
    country_list,rf_pred = final_method(X_raw)
    score = final_method2(Y_raw,rf_pred)
    result = jsonify({'prediction': country_list},{'score': score})
    #except Exception as e:
        #return render_template("index.html", error = str(e))
    return render_template('index.html',result = country_list,score = score)
    #return jsonify({'prediction': country_list},{'score': score})
        


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

    
