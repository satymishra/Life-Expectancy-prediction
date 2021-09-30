# importing files
import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import pickle 
import matplotlib.pyplot as plt


# Reading file
df=pd.read_csv('Life Expectancy Data.csv')


# Replacing all the null with mean of respected columns
column=df['Life expectancy ']
column.fillna(column.mean(),inplace=True)

column=df['Adult Mortality']
column.fillna(column.mean(),inplace=True)

column=df['Alcohol']
column.fillna(column.mean(),inplace=True)

column=df['Hepatitis B']
column.fillna(column.mean(),inplace=True)

column=df[' BMI ']
column.fillna(column.mean(),inplace=True)

column=df['Polio']
column.fillna(column.mean(),inplace=True)

column=df['Total expenditure']
column.fillna(column.mean(),inplace=True)

column=df['Diphtheria ']
column.fillna(column.mean(),inplace=True)

column=df['GDP']
column.fillna(column.mean(),inplace=True)

column=df['Population']
column.fillna(column.mean(),inplace=True)

column=df[' thinness  1-19 years']
column.fillna(column.mean(),inplace=True)

column=df[' thinness 5-9 years']
column.fillna(column.mean(),inplace=True)

column=df['Income composition of resources']
column.fillna(column.mean(),inplace=True)

column=df['Schooling']
column.fillna(column.mean(),inplace=True)


# converting country name to lower case
df['Country'] = df['Country'].str.lower()


# asking for the country
#cntri=input("Enter country name :")
#cntri=cntri.lower()
#country=df.loc[df['Country'] == f'{cntri}']


# assigning X and Y
#X=country[['Adult Mortality','Alcohol','Polio','Total expenditure','Diphtheria ',' HIV/AIDS','Income composition of resources','Schooling']]
#Y=country['Life expectancy ']

X=df[['Adult Mortality','Alcohol','Polio','Total expenditure','Diphtheria ',' HIV/AIDS','Income composition of resources','Schooling']]
Y=df['Life expectancy ']



# setting linear regression model
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X,Y)


# Saving model to disk
pickle.dump(lr,open('model.pkl','wb'))

# loading model 
model=pickle.load(open('model.pkl','rb'))


# predicting 
#print(model.predict([[56.0,67,58,8.16,8,0.1,.479,1]]))

