import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('Processed_Churn_data.csv')
dep_var = data.iloc[:, 0:15]
churn = data['Exited']

# Creating Train and Test Set
X_train, X_val, y_train, y_val = train_test_split(dep_var, churn, random_state = 0)

# Creating Random Forest Model
model_pi = RandomForestClassifier(random_state=1, n_estimators = 2000)
model_pi = model_pi.fit(X_train[['Age', 'NumOfProducts', 'EstimatedSalary', 'CreditScore', 'Balance', 'Tenure']],
 y_train)
preds_mpi = model_pi.predict(X_val[['Age', 'NumOfProducts', 'EstimatedSalary', 'CreditScore', 'Balance', 'Tenure']])
