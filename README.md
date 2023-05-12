# DS-EXERCISE-7                                           
# Feature-Selection

## AIM

To Perform the various feature selection techniques on a dataset and save the data to a file.

## Explanation

Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

## ALGORITHM

STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature selection techniques to all the features of the data set

STEP 4
Save the data to the file

## CODE

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df.head()

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df.head()

from sklearn.preprocessing import OrdinalEncoder

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df.head()

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df.head()

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer 

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 

y = df1["Survived"] 

plt.figure(figsize=(7,6))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

    p= [] 
    
    X_1 = X[cols]
    
    X_1 = sm.add_constant(X_1)
   
    model = sm.OLS(y,X_1).fit()
    
    p = pd.Series(model.pvalues.values[1:],index = cols)   
    
    pmax = max(p)
    
    feature_with_p_max = p.idxmax()
    
    if(pmax>0.05):
    
        cols.remove(feature_with_p_max)
        
    else:
    
        break
        
selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_) 

nof_list=np.arange(1,6)   

high_score=0

nof=0      

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    
    model = LinearRegression()
    
    rfe = RFE(model,step=nof_list[n])
    
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    
    X_test_rfe = rfe.transform(X_test)
    
    model.fit(X_train_rfe,y_train)
    
    score = model.score(X_test_rfe,y_test)
    
    score_list.append(score)
    
    if(score>high_score):
    
        high_score = score
        
        nof = nof_list[n]
        
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)   

X_rfe = rfe.fit_transform(X,y) 

model.fit(X_rfe,y)    

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()      

## OUTPUT

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/b24dc035-6447-4e9f-9506-8ec608fe64d7)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/406e94f1-7cb3-4c33-8348-b239e581817b)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/216830fa-c25a-4510-b557-fd84b4b2c407)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/640024e2-c8f4-4882-8f2e-165cc3b0801d)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/e33b94a1-80aa-4a68-938a-ac3a47ff1750)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/09d5f6e1-577a-4152-82cb-5650050978b5)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/ccfabfe2-976a-490b-be95-5000665e9a62)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/5829f8be-5f4e-4e7b-ac7d-2ddf02ef6bae)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/505937d4-79d0-4d0f-b426-eb96c5ff8284)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/bb8cb530-c4b9-4fd3-b6de-e547f9576fcd)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/74fdc5f4-6948-4238-bfc2-45f3dd865645)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/96c12e18-2b40-4495-9294-dfd6a3b33ae2)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/8fe5acae-e8a4-4dbd-997f-03593fb843de)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/22ddaa16-64ce-4bb5-acfd-286795dcdc7c)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/fdbf03b2-77e1-4602-ae2c-26113b322d50)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/f887a4f2-10a8-489e-874f-3238d6edef84)

![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-7/assets/126390051/9deae22d-402b-449f-8a59-e0ab7b708c1c)

## RESULT:

Thus the Feature Selection for the given datasets had been executed successfully




