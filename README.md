# Airbnb-New-User-Prediction
Kaggle Competition : 

AIRBNB NEW USER BOOKING PREDICTION PROBLEM :
Business Problem :
Airbnb has become a very popular choice among the travellers around the world for the kind of unique experiences that they provide and also for presenting an alternative to costly hotels. It is currently present in more than 34000+cities across 190 countries. Customers can make their bookings either through the website application or using the iOS/Android application. Airbnb is consistently trying to improve this booking experience to make it easier for the first time customer.

Problem Statement :
The problem that this case study is dealing with predicts the location that a user is most likely to book for the first time.

The accurate prediction helps to decrease the average time required to book by sharing more personalized recommendations and also in better forecasting of the demand. We use the browser’s session data as well as the user’s demographic information that is provided to us to create features that help in solving the problem.

Mapping the real world problem as a ML problem:
This is a multi-class classification problem where given the user data we have to predict top five most probable destinations among any of the 12 choices -US’, ‘FR’, ‘CA’, ‘GB’, ‘ES’,‘IT’, ‘PT’, ‘NL’,’DE’, ‘AU’, ‘NDF’ and ‘others’. ‘NDF’ and ‘others’ are different from each other. ‘NDF’ means there has been no booking done for this user and ‘others’ means that there has been a booking but to a country not present in the list given.

Data set Analysis: (Data downloaded from the Kaggle : https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/data)

1) Files given: train_users, test_users, sessions, countries, and age_gender_bkts.

2) Total File Size : 64.71MB

3) Total number of records: 2,13,451 (train_users), 62,096 (test_users)

4) The first two files contain the individual information of the users i.e. age, gender, signup_method, language, country_destination (target), etc.

5) The sessions.csv contains web session data of the users. Each record in this dataset is identified with user_id field that is corresponding to the id field of the train datasets. We find several session records containing information from the different times the particular user has accessed the Airbnb application.

6) The sessions.csv has data of users from 2014 onwards whereas the train dataset has records dating back to 2010.

7) The last two datasets contain more general statistical information of the destination and the users’ respectively.

Real World Business Constraints:

a) Low latency is important.

b) Mis-classification cost is not considerably high as the user can very easily change the destinations if he/she doesn’t like the given recommendations.

c) Interpretability of the result is not much needed.

Performance Metric:

NDCG (Normalized discounted cumulative gain) as required by the Kaggle competition.

Read the blog at : https://namrata-thakur893.medium.com/airbnb-new-user-prediction-a-kaggle-case-study-e26e712fe8d2
