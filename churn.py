# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure

data = pd.read_csv('Churn_Modelling.csv')

data

tenure = []
for value in data['Tenure']:
    if((value >= 0) and (value < 3)):
        value = 'Low'
        tenure.append(value)
    elif((value >= 3) and (value <= 6)):
        value = 'Moderate'
        tenure.append(value)
    else:
        value = 'High'
        tenure.append(value)

figure(figsize = (12, 5))
sns.histplot(data, x = 'EstimatedSalary')

data['EstimatedSalary'].quantile([0.25, 0.75])

sal_class = []
for value in data['EstimatedSalary']:
    if(value <= 50000):
        value = 'Low'
        sal_class.append(value)
    elif((value > 50000) and (value <= 150000)):
        value = 'Moderate'
        sal_class.append(value)
    else:
        value = 'High'
        sal_class.append(value)

tenure = pd.Series(tenure)

sal_class = pd.Series(sal_class)

data = pd.concat([data, sal_class.rename('sal_class')], axis = 1)

data = pd.concat([data, tenure.rename('tenure_class')], axis = 1)

data['Est_sal_ratio'] = data['EstimatedSalary']/data['EstimatedSalary'].median()

data

data['tenure_class'].head(10)

mod = data.iloc[2]
data.iloc[2] = data.iloc[7]
data.iloc[7] = mod

print(data['tenure_class'].head())
print("-------------------------")
print(data['sal_class'].head())

data.head()

data.iloc[1], data.iloc[7] = data.iloc[7], data.iloc[1]
data.iloc[0], data.iloc[16] = data.iloc[16], data.iloc[0]

data.iloc[2], data.iloc[5] = data.iloc[5], data.iloc[2]

data.head()

# Checking for null values:
for col in data.columns:
    print(f'{col}: {(data[col].loc[data[col].isnull() == True].shape[0])/data[col].shape[0]*100}%')

data_copy = pd.read_csv('Churn_Modelling.csv')

sns.boxplot(data = data_copy, x = 'CreditScore')

#pip install seaborn_qqplot
from seaborn_qqplot import pplot
from scipy.stats import norm
fig, axes_1 = plt.subplots(1,2, figsize = (12, 5))
plt.subplots_adjust(wspace=0.30, hspace=0.50)
sns.kdeplot(data = data_copy, x = 'CreditScore', fill = True, ax = axes_1[0])
sns.boxplot(data = data, x = 'Balance', ax = axes_1[1])
pplot(data = data_copy, x = 'CreditScore' , y = norm,kind = 'qq', height = 4, aspect = 2, display_kws = {'identity':True})

data_copy.head()

data_copy = data_copy.drop(data_copy.columns[:3], axis = 1)
data_copy.head()

iqr = data_copy['CreditScore'].quantile(0.75) - data_copy['CreditScore'].quantile(0.25)
cred_outliers = data_copy['CreditScore'][data_copy['CreditScore'] < data_copy['CreditScore'].quantile(0.25) - (1.5*iqr)]
cred_no_out = data_copy['CreditScore'][data_copy['CreditScore'] < 385]

sns.boxplot(data = data_copy, x = 'CreditScore')

data_copy = data_copy.drop(cred_outliers.index, axis = 0)

from seaborn_qqplot import pplot
from scipy.stats import norm
fig, axes_1 = plt.subplots(1,2, figsize = (12, 5))
plt.subplots_adjust(wspace=0.30, hspace=0.50)
sns.kdeplot(data = data_copy, x = 'CreditScore', fill = True, ax = axes_1[0])
sns.boxplot(data = data, x = 'Balance', ax = axes_1[1])
pplot(data = data_copy, x = 'CreditScore' , y = norm,kind = 'qq', height = 4, aspect = 2, display_kws = {'identity':True})

"""## Analysis

<h2> For the analysis we first see that we have data from three different nations: Germany, Spain, France.<br>
Due to the cultural differences, it makes sense that we make conclusions for each nation instead of making any conclusion on the whole dataset without taking into the account the geography</h2>
"""

import plotly.express as px
px.histogram(data, x = data['Geography'], color = data['tenure_class'])

import plotly.express as px
px.histogram(data, x = data['Geography'], color = data['Exited'] )

"""1. Out of the total churned customers, 32% German and 16% French Customers have churned, and nearly 17% Spanish Customers have churned.<br>
So German customers don't seem to be satisfied with the services

"""

px.histogram(data, x = data['Geography'], color = data['sal_class'] )

temp_1 = data[data['Exited'] == 1]
temp_2 = data[data['Exited'] == 0]

px.histogram(temp_1, x = temp_1['Geography'], color = temp_1['sal_class'], title = 'Salary Class in different regions among the customers who Churned' )

px.histogram(temp_2, x = temp_2['Geography'], color = temp_2['sal_class'], title = 'Salary Class in different regions among the customers who not Churned' )

data_ger = data.loc[data['Geography'] == 'Germany']
data_fra = data.loc[data['Geography'] == 'France']
data_esp = data.loc[data['Geography'] == 'Spain']

for clas in sal_class.unique():
    print(f"In Germany {clas}: {data_ger.loc[(data_ger['sal_class'] == clas) & (data_ger['Exited'] == 1)].shape[0]/data_ger.loc[(data_ger['sal_class'] == clas)].shape[0]*100}")
    print(f"In France {clas}: {data_fra.loc[(data_fra['sal_class'] == clas) & (data_fra['Exited'] == 1)].shape[0]/data_fra.loc[(data_fra['sal_class'] == clas)].shape[0]*100}")
    print(f"In Spain {clas}: {data_esp.loc[(data_esp['sal_class'] == clas) & (data_esp['Exited'] == 1)].shape[0]/data_esp.loc[(data_esp['sal_class'] == clas)].shape[0]*100}")
    print('-'*50)

"""2. In France and Spain, the customer churn distributed by Salary Class is less than 20% of their respective class, but in Germany it is greater than 30%, high proportion of churning among the high Salary class customers show prices of company products may not be the only reason for the Churn"""

data_ger_cred = data.drop(cred_outliers.index, axis = 0)
data_ger_cred = data_ger_cred.loc[data_ger_cred['Geography'] == 'Germany']

data_fra_cred = data.drop(cred_outliers.index, axis = 0)
data_fra_cred = data_fra_cred.loc[data_fra_cred['Geography'] == 'France']

data_esp_cred = data.drop(cred_outliers.index, axis = 0)
data_esp_cred = data_esp_cred.loc[data_esp_cred['Geography'] == 'Spain']

fig, axes = plt.subplots(3,2, figsize = (15,12))
sns.boxplot(data = data_ger_cred.loc[data_ger_cred['Exited'] == 1], x = 'sal_class', y = 'CreditScore', ax = axes[0][0], order = ['Low', 'Moderate', 'High'])
sns.boxplot(data = data_ger_cred.loc[data_ger_cred['Exited'] == 0], x = 'sal_class', y = 'CreditScore', ax = axes[0][1], order = ['Low', 'Moderate', 'High'])
axes[0][0].title.set_text('Germany, Churned')
axes[0][1].title.set_text('Germany, Not Churned')

sns.boxplot(data = data_fra_cred.loc[data_fra_cred['Exited'] == 1], x = 'sal_class', y = 'CreditScore', ax = axes[1][0], order = ['Low', 'Moderate', 'High'])
sns.boxplot(data = data_fra_cred.loc[data_fra_cred['Exited'] == 0], x = 'sal_class', y = 'CreditScore', ax = axes[1][1], order = ['Low', 'Moderate', 'High'])
axes[1][0].title.set_text('France, Churned')
axes[1][1].title.set_text('France, Not Churned')

sns.boxplot(data = data_esp_cred.loc[data_esp_cred['Exited'] == 1], x = 'sal_class', y = 'CreditScore', ax = axes[2][0], order = ['Low', 'Moderate', 'High'])
sns.boxplot(data = data_esp_cred.loc[data_esp_cred['Exited'] == 0], x = 'sal_class', y = 'CreditScore', ax = axes[2][1], order = ['Low', 'Moderate', 'High'])
axes[2][0].title.set_text('Spain, Churned')
axes[2][1].title.set_text('Spain, Not Churned')
plt.subplots_adjust(wspace=0.20, hspace=0.40)

"""3. Across all the nations, most of the customers have similar proportion of customers with good and bad for all the salary class"""

exit_0 = data['Exited'][data['Exited'] == 0]
exit_1 = data['Exited'][data['Exited'] == 1]

fig, axes = plt.subplots(1,2, figsize = (15, 6))
sns.histplot(data = data, x = 'Age', hue = exit_1, ax = axes[0])
sns.histplot(data, x = 'Age', hue = exit_1, ax = axes[1], cumulative = True)
plt.subplots_adjust(wspace=0.30, hspace=0.30)

# Exited vs Estimated Salary
# CreditScore vs Balance

"""4. We see that customers from the age group 40 - 50 constitute 43% of the total number of churns. <br>
So we can create provide some services directed at these age groups so they don't leave.
"""

fig, axes = plt.subplots(3,2, figsize = (15,12))
sns.boxplot(data = data_ger_cred.loc[data_ger_cred['Exited'] == 1], x = 'sal_class', y = 'CreditScore', ax = axes[0][0], order = ['Low', 'Moderate', 'High'])
sns.boxplot(data = data_ger_cred.loc[data_ger_cred['Exited'] == 0], x = 'sal_class', y = 'CreditScore', ax = axes[0][1], order = ['Low', 'Moderate', 'High'])
axes[0][0].title.set_text('Germany, Churned')
axes[0][1].title.set_text('Germany, Not Churned')

sns.boxplot(data = data_fra_cred.loc[data_fra_cred['Exited'] == 1], x = 'sal_class', y = 'CreditScore', ax = axes[1][0], order = ['Low', 'Moderate', 'High'])
sns.boxplot(data = data_fra_cred.loc[data_fra_cred['Exited'] == 0], x = 'sal_class', y = 'CreditScore', ax = axes[1][1], order = ['Low', 'Moderate', 'High'])
axes[1][0].title.set_text('France, Churned')
axes[1][1].title.set_text('France, Not Churned')

sns.boxplot(data = data_esp_cred.loc[data_esp_cred['Exited'] == 1], x = 'sal_class', y = 'CreditScore', ax = axes[2][0], order = ['Low', 'Moderate', 'High'])
sns.boxplot(data = data_esp_cred.loc[data_esp_cred['Exited'] == 0], x = 'sal_class', y = 'CreditScore', ax = axes[2][1], order = ['Low', 'Moderate', 'High'])
axes[2][0].title.set_text('Spain, Churned')
axes[2][1].title.set_text('Spain, Not Churned')
plt.subplots_adjust(wspace=0.20, hspace=0.40)

"""5. In all the countries we see that out of those customers that churned, most of them belong to the age group 40 - 50 years, so the services of the bank might not be attractive to those age group, on the other hand customers who stayed mostly belong to the age group 20 - 40"""

data.columns

temp = data.loc[(data['Age'] > 39) & (data['Age'] < 51)]
(temp.Exited[temp.Exited == 1].count()/data.Exited[data.Exited == 1].shape[0])*100



"""<h2> Data Preparation for Models</h2>"""

mod_data =  data_copy.drop(['Exited'], axis = 1)
target = data_copy['Exited']

pd.Series(data_copy.columns)

cat_indices = [1,2,7,8]
from imblearn.over_sampling import SMOTENC
X = mod_data
y = target
smote_nc = SMOTENC(categorical_features = cat_indices, random_state = 0)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)

fig_up, ax_up = plt.subplots(1, 2, figsize = (10,5))
target_count = data_copy['Exited'].value_counts()
target_count_2 = pd.DataFrame(y_resampled).value_counts()
target_count.plot(kind = 'bar', title = 'Before Upsampling', ax = ax_up[0])
target_count_2.plot(kind = 'bar',title = 'After Upsampling', ax = ax_up[1])
plt.subplots_adjust(wspace=0.50, hspace=0.30)

X_resampled = pd.DataFrame(X_resampled, columns = mod_data.columns)
y_resampled = pd.DataFrame(y_resampled)
y_resampled.columns = ['Exited']

y_resampled

#OHE Encoding
data_ohe =  pd.get_dummies(X_resampled, columns = ['Gender', 'Geography', 'IsActiveMember', 'HasCrCard'])
data_ohe = pd.concat([data_ohe, y_resampled], axis = 1)

data_ohe.columns[:6]

data_ohe_copy = data_ohe.copy()

colnames = data_ohe.columns[:6]
features = data_ohe_copy[colnames]

# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features.values)

data_ohe_copy[colnames] = scaled_features

data_ohe_copy.head()

dep_var = data_ohe_copy.iloc[:, 0:15]
churn = data_ohe_copy['Exited']

# Creating Train and Test Set
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(dep_var, churn, random_state = 0)

"""<h2> Logistic Regression </h2>"""

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
log_reg = sm.Logit(y_train, X_train).fit()

print(log_reg.summary())

from sklearn.metrics import accuracy_score, confusion_matrix
pred_logistic = log_reg.predict(X_val)
prediction_logistic = list(map(round, pred_logistic))

cm = confusion_matrix(y_val, prediction_logistic) 
print("Confusion Matrix: \n", cm)
print('Accuracy on test_set: ', accuracy_score(y_val, prediction_logistic)*100)

from sklearn.model_selection import cross_val_score
X_cv = dep_var
y_cv = churn
model_cv = LogisticRegression()
cross_val = cross_val_score(model_cv, X_cv, y_cv, scoring='accuracy')
print(cross_val)
print(cross_val.mean()*100)

"""<h2> Random Forest </h2>"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

model_rf = RandomForestClassifier(random_state = 1)
model_rf.fit(X_train, y_train)
preds_rf = model_rf.predict(X_val)
cm_rf = confusion_matrix(y_val, preds_rf) 
print("Confusion Matrix: \n", cm_rf)
print("Accuracy on validation set: ", accuracy_score(preds_rf, y_val))

from sklearn.model_selection import cross_val_score
X_cv_rf = dep_var
y_cv_rf = churn
model_cv_rf = RandomForestClassifier()
cross_val = cross_val_score(model_cv_rf, X_cv_rf, y_cv_rf, scoring='accuracy')
print(cross_val)
print(cross_val.mean()*100)
"""<h2>To get the output many features have to be given in the input.<br>
We wish to reduce the number of features to be given in the input to the model with little to no reduce in the Accuracy of the model</h2>"""

"""<h2> Important features </h2>"""
plt.barh(X_train.columns, model_rf.feature_importances_)

from sklearn.model_selection import cross_val_score
X_cv_rf = dep_var[['Age', 'NumOfProducts', 'EstimatedSalary', 'CreditScore', 'Balance', 'Tenure']]
y_cv_rf = churn
model_cv_rf = RandomForestClassifier(random_state=1)
cross_val = cross_val_score(model_cv_rf, X_cv_rf, y_cv_rf, scoring='accuracy')
print(cross_val)
print(cross_val.mean()*100)

X_cv_rf = dep_var[['Age', 'NumOfProducts', 'EstimatedSalary', 'CreditScore', 'Balance', 'Tenure']]
y_cv_rf = churn
'''estimates = [100, 500, 1000]

for n in estimates:
    model_cv_rf = RandomForestClassifier(random_state = 1, n_estimators = n)
    cross_val = cross_val_score(model_cv_rf, X_cv_rf, y_cv_rf, scoring='accuracy')
    print(cross_val)
    print(cross_val.mean()*100)
    print('--------------------\n')'''
    
model_pi = RandomForestClassifier(random_state=1, n_estimators = 1000)
model_pi = model_pi.fit(X_train[['Age', 'NumOfProducts', 'EstimatedSalary', 'CreditScore', 'Balance', 'Tenure']], y_train)

preds_mpi = model_pi.predict(X_val[['Age', 'NumOfProducts', 'EstimatedSalary', 'CreditScore', 'Balance', 'Tenure']])
print("Accuracy on validation set: ", accuracy_score(preds_mpi, y_val))

'''import pickle
 #Storing the model in a pickle file
pickle.dump(model_pi, open('rfmodel.pkl', 'wb'))
model = pickle.load(open('rfmodel.pkl', 'rb'))'''


"""<h2>Conclusions:<br></h2>

1. Out of the total churned customers, 32% German and 16% French Customers have churned, and nearly 17% Spanish Customers have churned.<br>
So German customers don't seem to be satisfied with the services<br>

2. In France and Spain, the customer churn distributed by Salary Class is less than 20% of their respective class, but in Germany it is greater than 30%, high proportion of churning among the high Salary class customers show prices of company products may not be the only reason for the Churn<br>

3. We see that customers from the age group 40 - 50 constitute 43% of the total number of churns.
So we can create provide some services directed at these age groups so they don't leave.<br>

4. In all the countries we see that out of those customers that churned, most of them belong to the age group 40 - 50 years, so the services of the bank might not be attractive to those age group, on the other hand customers who stayed mostly belong to the age group 20 - 40<br>

5. Among those who churned we see that proportion of customers with low credit score is relatively high than the other classes across different salary classes. But among those who didn't also we can see good proportion of customers with lower credit score

"""
