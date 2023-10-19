#!/usr/bin/env python
# coding: utf-8

# In[295]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import preprocessing
from sklearn import utils

import datetime as dt


# In[296]:


df = pd.read_csv('newairquality.csv', parse_dates=[['Date', 'Time']])


# In[297]:


df.head()


# In[ ]:





# In[391]:


df['Date_Time'] = pd.to_datetime(df['Date_Time'])
df['Date_Time'] = df['Date_Time'].apply(lambda x: x.value) # converting date time to numerical value

# will be using Date_Time column instead 
X = df.drop(['T','Month','Day','Year'], axis=1) 
y = df['T']

df.head()


# In[392]:


#Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression, chi2


uni = SelectKBest(score_func = f_regression, k = 10)
fit = uni.fit(X, y)

X.columns[fit.get_support(indices=True)].tolist()


# In[ ]:





# In[393]:


# Splitting Data into Train and Test

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


# # Linear Regression

# In[394]:


# But I am having a lot of issues trying to plot the chemicals over datetime.  
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)


# In[395]:


Linear_train_prediction = linear_model.predict(X_train)


# In[396]:


print(Linear_train_prediction)


# In[397]:


Linear_test_prediction = linear_model.predict(X_test)
print(Linear_test_prediction)


# In[398]:


plt.scatter(y_test, Linear_test_prediction)
plt.xlabel("Actual Value")
plt.ylabel("Predict Value")
plt.show()


# In[ ]:





# # SVM Regression

# In[399]:


from sklearn.svm import SVR


# In[400]:


df.head()


# In[401]:


# Split Data 
print(X.shape, X_train.shape, X_test.shape)


# In[411]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

svm_X_train = sc.fit_transform(X_train)
svm_X_test = sc.fit_transform(X_test)


SVR_Regression = SVR(kernel='rbf').fit(svm_X_train,y_train)


# In[412]:


SVR_score = SVR_Regression.score(svm_X_train,y_train)
print(SVR_score)


# In[413]:


#Prediction on training data
SVR_train_prediction = SVR_Regression.predict(svm_X_train)
print(SVR_train_prediction)


# In[ ]:





# In[414]:


#Prediction on testing data
SVR_test_prediction = SVR_Regression.predict(svm_X_test)
print(SVR_test_prediction)


# In[415]:


plt.scatter(y_test, SVR_test_prediction)
plt.xlabel("Actual Value")
plt.ylabel("Prediction Value")
plt.show()


# # KNN Regression

# In[416]:


from sklearn.neighbors import KNeighborsClassifier


# In[417]:


# Knn Model with k = 5
KNN_Regression = KNeighborsClassifier(n_neighbors = 3)


# In[418]:


# train the model
KNN_Regression.fit(X_train, y_train.astype('int'))


# In[419]:


#Prediction on training data
KNN_data_prediction = KNN_Regression.predict(X_train)
print(KNN_data_prediction)


# In[420]:


#Prediction on testing data
KNN_test_prediction = KNN_Regression.predict(X_test)
print(KNN_test_prediction)


# In[421]:


plt.scatter(y_test, KNN_test_prediction)
plt.xlabel("Actual Value")
plt.ylabel("Prediction Value")
plt.show()


# In[ ]:





# # Decision Tree Regression
# 

# In[422]:


from sklearn.tree import DecisionTreeRegressor 


# In[423]:


RandomTree_Regression = DecisionTreeRegressor(random_state=0)


# In[424]:


#Fitting Model 
RandomTree_Regression.fit(X_train, y_train)

#Prediction on training data
RT_train_prediction = RandomTree_Regression.predict(X_train)
print(RT_train_prediction)

#Prediction on test data
RT_test_prediction = RandomTree_Regression.predict(X_test)
print(RT_test_prediction)


# In[425]:


plt.scatter(y_test, RT_test_prediction)
plt.xlabel("Actual Value")
plt.ylabel("Prediction Value")
plt.show()


# # Evaluation and Analysis 
# 
# 

# In[426]:


#Exploring Data ( Month Vs Temperature)

x1 = df['T']
y1 = df['Month']
plt.scatter(y1, x1)
plt.xlabel("Month")
plt.ylabel("Tempearture")

plt.show()


# 
# To determine whether pollutants affect the weather/temperature, we can plot a graph showing Temperature for each month.  
# From the month of May through October, we see the hottest temperature in the city reaching 40 Celcius(104 F) above.  
# This is also mainly because it is summer in Italy. It is still uncertain whether pollutants are a cause. 
# The most infamous pollutant that causes global warming is Carbon Monoxide. We can plot that in the next graph
# 

# In[427]:


#Exploring Data ( Co vs Month)

x1 = df['Month']
y1 = df['PT08.S2(NMHC)']

plt.scatter(x1, y1,color='y')
plt.ylabel("HydroCarbons")
plt.show()


# 
# This data seems to contradict what I previously had in mind. I thought CO gas was the main cause of global warming. 
# But it seems that CO levels are lowest during the months of May, June...September. 
# CO levels seem to spike from September to December, during the winter. 
# This makes sense if we consider heaters during the winter can produce carbmon monoxide
# 
# 

# In[428]:


#Exploring Data ( NO2 vs Month)

x1 = df['Month']
y1 = df['NO2(GT)']
z1 = df['NOx(GT)']

plt.scatter(x1, y1, color='r', label='NO2(GT)')
plt.xlabel("Month")
plt.ylabel("NO2(GT)")
plt.show()

plt.scatter(x1, z1, color='g', label='NOx(GT)')
plt.xlabel("Month")
plt.ylabel("NOx(GT)")
plt.show()



# It seems that Total Nitrogen Oxides (NOx) and Nitrogen Dioxdes also seems to spike during the months of Sept through Feb. It seems that farmers tend to use nitrogen application on plants to surive the winter. It may also be a reason why nitrogen spikes during winter.
# 
# 

# In[429]:


#Exploring Data ( Co vs Month)

x1 = df['Year']
y1 = df['CO(GT)']
z1 = df['T']
u1 = df['C6H6(GT)']

plt.scatter(x1, y1)
plt.xlabel("Year")
plt.ylabel("Carbon Monoxide")
plt.show()

plt.scatter(x1, u1,color='y')
plt.xlabel("Year")
plt.ylabel("Benzene")
plt.show()

plt.scatter(x1, z1,color='g')
plt.xlabel("Year")
plt.ylabel("Temperature")
plt.show()



# Overall temperature seems to be lower during 2005 than 2004. Coincidentally, polutants such as CO, Benzene, and Nitrogren groups also have lower levels recorded. 

# # Model Accuracy Evaluations

# In[430]:


#Linear Model Evaluation

#R2 error
linear_score = metrics.r2_score(y_test, Linear_test_prediction)

#mean absolute error
linear_MAE_score = metrics.mean_absolute_error(y_test, Linear_test_prediction)

#mean squared error
linear_MSE_score = metrics.mean_squared_error(y_test, Linear_test_prediction)

#root mean squared error 
linear_root_MSE_score = np.sqrt(linear_MSE_score)

print("Linear Model Evaluation")
print("R2 Score : ", linear_score)
print("Mean Absolute Error : ", linear_MAE_score)
print("Mean Squared Error : ", linear_MSE_score)
print("Root Mean Squared Error:", linear_root_MSE_score )


# In[431]:


#KNN Regression

#R2 Error
r2_score_knn = metrics.r2_score(y_test, KNN_test_prediction)

#mean absolute error
KNN_mean_absolute_error = metrics.mean_absolute_error(y_test, KNN_test_prediction)

#mean squared error
KNN_MSE_score = metrics.mean_squared_error(y_test, KNN_test_prediction)

#root mean squared error 
KNN_root_MSE_score = np.sqrt(KNN_MSE_score)


print("KNN Model Evaluation")
print("R2 Score : ", r2_score_knn)
print("Mean Absolute Error: ", KNN_mean_absolute_error)
print("Mean Squared Error: ", KNN_MSE_score)
print("Root Mean Squared Error:", KNN_root_MSE_score )


# In[434]:


#SVM Regression

#r2 Error
SVR_r2_score = metrics.r2_score(y_test, SVR_test_prediction)

#Mean Absolute Error
SVR_mean_absolute_error = metrics.mean_absolute_error(y_test, SVR_test_prediction)

#mean squared error
SVR_MSE_score = metrics.mean_squared_error(y_test, SVR_test_prediction)

#root mean squared error 
SVR_root_MSE_score = np.sqrt(SVR_MSE_score)


print("SVM Regression Model Evaluation")
print("R2 Score: ",SVR_r2_score)
print("SVR Mean Absolute Error:",SVR_mean_absolute_error)
print("SVR Squared Error:",SVR_MSE_score)
print("Root Mean Squared Error:", SVR_root_MSE_score )


# In[433]:


#Random Tree Regression

#r2 score
score_random_tree = metrics.r2_score(y_test, RT_test_prediction)

#mean absolute error
MAE_score_random_tree = metrics.mean_absolute_error(y_test, RT_test_prediction)

#mean squared error
MSE_score_random_tree = metrics.mean_squared_error(y_test, RT_test_prediction)

#root mean squared error 
RT_root_MSE_score = np.sqrt(MSE_score_random_tree)


print("Random Tree Regression")
print("R2 Score:", score_random_tree)
print("Mean Absolute Error:", MAE_score_random_tree)
print("Mean Absolute Error:", MSE_score_random_tree)
print("Root Mean Squared Error:", RT_root_MSE_score )




# In conclusion, the random tree regression seems to yield the highest result. It has an r2 score of 1.0 and mean absolute error of 0.5. Since random tree is less likey to overfit, I did not perfrom any cross validation on this model. The SVM regression peformed the worst having very high inaccurate scores. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




