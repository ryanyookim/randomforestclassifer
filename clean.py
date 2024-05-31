import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, classification_report

# Get the training data
train = pd.read_stata('H22J3_R.dta')

# This prints out the rows and columns
print("Rows and Colums")
print(train.shape)

# Print the first 5 rows of the data
print("First 5 rows")
print(train.head(5))

# How many nulls are there?
print("How many nulls are there?")
print(train.isnull().sum().sum())

# Replace nulls with 0 in the data
train = train.fillna(0)

# Output to CSV
train.to_csv('dataInitialClean.csv')

# How many nulls are there?
print("How many nulls are there now?")
print(train.isnull().sum().sum())
print(train.head(5))
# Replace nulls with 0 in the data
#train.fillna("novals", inplace=True)
#train.fillna(0, inplace=True)
#train.fillna(0)

# Not sure why this column still has blanks. Just in case make it a string
# train["SPN_SP"] = train["SPN_SP"].apply(str)


#for col in train.dtypes[train.dtypes == "object"].index:
#    for_dummy = train.pop(col)
#    train = pd.concat([train, pd.get_dummies(for_dummy, prefix=col)], axis=1)
#print(train.head())

# Print
# print("How many nulls thare there now")
# print(train.isnull().sum().sum())
# print(train)

# labels = train.pop("SJ3584")
# print(labels)

# x_train, x_test, y_train, y_test = train_test_split(train, labels, test_size=0.25)
# rf = RandomForestClassifier()
# rf.fit(x_train, y_train)

# y_pred = rf.predict(x_test)

# print("Pred=")
# print(y_pred)

# print(rf.feature_importances_)

# print(confusion_matrix(y_test, y_pred))

# print(accuracy_score(y_test, y_pred))

# print(classification_report(y_pred, y_test))
#train.to_csv('cleaned.csv')


#print(train)

#print(df.isnull().sum().sum())
#df.fillna(0)
#print(df.isnull().sum().sum())
#print(df)

# Find the number of nulls in the data
#NAs = pd.concat([train.isnull().sum()], axis=1, keys=["Nulls"])
#NAs[NAs.sum(axis=1) > 0]
#print(NAs)



#nan_rows = train[train['SPN_SP'].isnull()]
#print(nan_rows)


#msg = "hello world"
#print(msg)


#df = pd.read_csv('data.csv')

#print(df.to_string()) 

#data = pd.read_stata('h20f1a.dta')
#data.to_csv('h20.csv')

#data = pd.read_stata('H22J3_R.dta')
#data.to_csv('retirement.csv')

#print (data.to_string())