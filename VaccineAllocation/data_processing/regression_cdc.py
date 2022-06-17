import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Data Processing
# data = pd.read_csv("/Users/kevinli/Documents/Regression dataset.csv")
# feature=data.drop(['Date','Symptomatic', '7-day new case sum / 100000', 
#                    'hospitalization/100k'],axis=1).values
# target = data['7-day new case sum / 100000'].values
# target=target[10:]
# feature=feature[10:]

## Linear Regression 

#Plotting
# plt.scatter(feature,target)
# plt.xlabel("Symptomatic per 100k", fontsize=12)
# plt.ylabel("7-day sum per 100k", fontsize=12)

# #Train, test split 
# train,test,train_label,test_label=train_test_split(feature,target,test_size=0.33,random_state=222)
# reg=LinearRegression(fit_intercept=True)
# model = reg.fit(train,train_label)
# predict = model.predict(test)
# print(r2_score(test_label,predict))


# #Logistics Regression
# for x, label in enumerate(target):
#       target[x] = 1 if (label >= 200) else 0
        
# plt.figure()
# fig=sns.regplot(feature, target, logistic=True)
# fig.set(xlabel='Symptomatic', ylabel='f(x)')
# model = LogisticRegression(random_state=0).fit(feature, target)
# print(model.score(feature, target))


# #Plot Confusion Matrix 
# cm = confusion_matrix(target, model.predict(feature))
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(cm)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# plt.title("Heat Map", fontsize=16)
# plt.show()

# --------------- 200+ cases only (+/- 3 days) --------

# Data Processing
data = pd.read_csv("/Users/kevinli/Documents/Regression dataset copy.csv")
feature=data.drop(['Date','Symptomatic', '7-day new case sum / 100000', 'hospitalization/100k'],axis=1).values
target =data['7-day new case sum / 100000'].values
# Removing past 40 days as outliers (because 7 day moving average is not accurate)


# keep = np.ones(target.shape, dtype=bool)
# keep2 = np.ones(feature.shape, dtype=bool)
# for pos, val in enumerate(target):
#     if val > 200 or val+1>200 or val+2>200 or val+3>200 or val+4>200 or val-1>200 or val-2>200:
#         keep[pos] = True
#         #keep2[pos]=False

# target = target[keep]

# feature=feature[keep]

# Linear Regression 

# Plotting
plt.scatter(feature,target)
plt.xlabel("Symptomatic per 100k", fontsize=12)
plt.ylabel("7-day sum per 100k", fontsize=12)

#Train, test split 
train,test,train_label,test_label=train_test_split(feature,target,test_size=0.33,random_state=222)
reg=LinearRegression(fit_intercept=True)
model = reg.fit(train,train_label)
predict = model.predict(test)
print(r2_score(test_label,predict))


#Logistics Regression
for x, label in enumerate(target):
      target[x] = 1 if (label >= 200) else 0
        
plt.figure()
fig=sns.regplot(feature, target, logistic=True)
fig.set(xlabel='Symptomatic', ylabel='f(x)')
model = LogisticRegression(random_state=0).fit(feature, target)
print(model.score(feature, target))


#Plot Confusion Matrix 
cm = confusion_matrix(target, model.predict(feature))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title("Heat Map", fontsize=16)
plt.show()

# --------------- With HOSPITALIZATION ADMISSIONS + 200 CASES + --------

data = pd.read_csv("/Users/kevinli/Documents/Regression dataset copy.csv")
feature=data.drop(['Date','Symptomatic', '7-day new case sum / 100000'],axis=1).values
target =data['7-day new case sum / 100000'].values
#Removing past 40 days as outliers (because 7 day moving average is not accurate)


#Linear Regression 

#Train, test split 
train,test,train_label,test_label=train_test_split(feature,target,test_size=0.33,random_state=222)
reg=LinearRegression(fit_intercept=True)
model = reg.fit(train,train_label)
predict = model.predict(test)
coef = reg.coef_
intercept = reg.intercept_
print("Linear reg score")
print(r2_score(test_label,predict))

plt.scatter(target, reg.predict(feature))
plt.xlabel('Case From Dataset')
plt.ylabel('Case Predicted By Model')
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
plt.title("Linear Regression with Hospitalization")

#try the model predictors
today=reg.predict(feature)


#Logistics Regression
for x, label in enumerate(target):
      target[x] = 1 if (label >= 200) else 0
        
model = LogisticRegression(random_state=0).fit(feature, target)
print("Logistic reg score")
print(model.score(feature, target))


#Plot Confusion Matrix 
cm = confusion_matrix(target, model.predict(feature))
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title("Heat Map", fontsize=16)
plt.show()

