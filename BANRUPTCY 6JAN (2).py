#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import scikitplot as skplt
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


# In[30]:


df = pd.read_csv('bankruptdata.csv')


# In[31]:


df


# In[32]:


shape= df.shape()
print(shape)


# In[ ]:


_, axes = plt.subplots(6,5, figsize=(15, 15))
not_bankrupt = [df.Bankrupt== 0]
is_bankrupt = [df.Bankrupt== 1]
ax = axes.ravel()                     # flatten the 2D array
for i in range(95):                   # for each of the 30 features
    bins = 20
    #---plot histogram for each feature---
    ax[i].hist(not_bankrupt[:i], bins=bins, color='r', alpha=.5)
    ax[i].hist(is_bankrupt[:i], bins=bins, color='b', alpha=0.3)
    #---set the title---
    #---display the legend---
    ax[i].legend(['not_bankrupt','is_bankrupt'], loc='best', fontsize=8)
    
plt.tight_layout()
plt.show()


# In[ ]:


df.info()


# In[ ]:


df['Bankrupt'].value_counts()


# In[ ]:


data = df["Bankrupt"].value_counts()
plt.pie(data,autopct='%1.2f%%',labels=data.index)
plt.show()


# In[ ]:


X = df.copy() 

y = df['Bankrupt']

X = X.drop(['Bankrupt'], axis=1)


# # Check for collinearity

# In[ ]:


df_corr = df.corr()['Bankrupt'].abs().sort_values(ascending=False)
df_corr


# In[ ]:


# get all the features that has at least 0.1 in correlation to the 
# target
features = df_corr[df_corr > 0.1].index.to_list()[1:]
len(features)                         


# # Multicolinearity: VIF >5 means high multi collinearity

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression 
def calculate_vif(df, features):    
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]        
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)                
        
        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})
calculate_vif(df,features)


# In[ ]:


df2 = calculate_vif(df, features)
df2


# In[ ]:


calculate_vif(df, features)['VIF'].sort_values()




# In[ ]:


features_to_remove = df2.loc[df2['VIF'] < 5]
features_to_remove


# In[ ]:





# In[ ]:


X = df.copy() 

y = df['Bankrupt']

X = X.drop(['Bankrupt'], axis=1)


# In[ ]:


X.columns


# In[ ]:


X


# In[ ]:


import numpy as np
var = np.var(X)


# In[ ]:


print(var)


# In[ ]:





# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[37]:


X_train.shape, X_test.shape


# In[38]:


#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[71]:


#1. Logestic Regression
from sklearn.linear_model import LogisticRegression
lgclassifier = LogisticRegression(random_state = 42)
lgclassifier.fit(X_train, y_train)


# In[ ]:


y_pred = lgclassifier.predict(X_test)
print(y_pred)


# In[ ]:


y_pred = lgclassifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


c_matrix = confusion_matrix(y_test,y_pred)
#print confusion matrix
print(c_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,display_labels=["Not bankrupt", "Is bankrupt"])
disp.plot()
plt.show()


# In[ ]:


y_val_scores = lgclassifier.predict_proba(X_test)
print(y_val_scores)

y_train_scores = lgclassifier.predict_proba(X_train)
print(y_train_scores)


# In[ ]:


# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_val_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[ ]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_val_scores[:,1])
print('AUC: ' + str(auc))


# In[ ]:


sns.heatmap(cm,annot=True)
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_test, y_pred)
print("Accuracy score for Logistic Regression Model: {:.2f} %".format(acc1*100))


# In[ ]:


predictions_val = lgclassifier.predict(X_test)
predictions_train = lgclassifier.predict(X_train)


# In[ ]:


from sklearn. metrics import classification_report

print('Validation Data Classification Report \n', classification_report(y_test, predictions_val))
print('Train Data Classification Report \n', classification_report(y_train, predictions_train))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_val_scores[:,1])
print('AUC: ' + str(auc))


# In[ ]:


resultsLogisticRegression = pd.DataFrame({'Train Accuracy': accuracy_score(y_train, predictions_train),
              'Test Accuracy': accuracy_score(y_test, predictions_val),
              'Train F1 Score':f1_score(y_train, predictions_train),
              'Test F1 Score':f1_score(y_test, predictions_val),
              'Train Precision':precision_score(y_train, predictions_train),
              'Test Precision': precision_score(y_test, predictions_val),
              'Train Recall':recall_score(y_train, predictions_train),
              'Test Recall': recall_score(y_test, predictions_val),
              'ROC AUC':auc},
             index=['LogisticRegression'])
resultsLogisticRegression


# In[ ]:


#K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


sns.heatmap(cm,annot=True)
plt.show()


# In[ ]:


acc3 = accuracy_score(y_test, y_pred)
print("Best Accuracy of K-NN: {:.2f} %".format(acc3*100))


# In[ ]:


test_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred_test=knn_model.predict(X_test)
    test_error = 1-accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)
print(test_error_rates.index(min(test_error_rates)))


# In[ ]:


print(classification_report(y_test,y_pred))


# In[ ]:


plt.plot(range(1,30), test_error_rates)
plt.ylabel("Error Rate")
plt.xlabel("K Neighbors")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc6 = accuracy_score(y_test, y_pred)


# In[ ]:


print(f"Random Forest Classification accuracy: {acc6}")


# # PCA

# In[ ]:


from sklearn.decomposition import PCA
 
pca = PCA(n_components = 2)
 
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
 
explained_variance = pca.explained_variance_ratio_
explained_variance


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 42)
classifier.fit(X_train, y_train)


# In[ ]:


y_pred = classifier.predict(X_test)
print(y_pred)


# In[ ]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[ ]:


c_matrix = confusion_matrix(y_test,y_pred)
#print confusion matrix
print(c_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,display_labels=["Not bankrupt", "Is bankrupt"])
disp.plot()
plt.show()


# In[ ]:


sns.heatmap(cm,annot=True)
plt.show()


# In[ ]:


from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_test, y_pred)
print("Accuracy score for Logistic Regression Model: {:.2f} %".format(acc1*100))


# In[ ]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, roc_auc_score


# In[ ]:


predictions_val = classifier.predict(X_test)
predictions_train = classifier.predict(X_train)


# In[ ]:


from sklearn. metrics import classification_report
print('Validation Data Classification Report \n', classification_report(y_test, predictions_val))


# In[ ]:


# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_val_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


# In[ ]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_val_scores[:,1])
print('AUC: ' + str(auc))


# In[ ]:


resultsLogisticRegression = pd.DataFrame({'Train Accuracy': accuracy_score(y_train, predictions_train),
              'Test Accuracy': accuracy_score(y_test, predictions_val),
              'Train F1 Score':f1_score(y_train, predictions_train),
              'Test F1 Score':f1_score(y_test, predictions_val),
              'Train Precision':precision_score(y_train, predictions_train),
              'Test Precision': precision_score(y_test, predictions_val),
              'Train Recall':recall_score(y_train, predictions_train),
              'Test Recall': recall_score(y_test, predictions_val),
              'ROC AUC':auc},
             index=['LogisticRegression'])
resultsLogisticRegression


# # FEATURE SELECTION

# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest


# In[52]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[53]:


mi = mutual_info_classif(X_train, y_train)
mi


# #the smaller the value of the mi, the less information we can infer from the feature about the target

# In[54]:


# the less mi value, the less information we can get from
mi = mutual_info_classif(X_train, y_train)
miseries = pd.Series(mi)
miseries
miseries.index = X_train.columns
miseries.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.ylabel('Mutual Information')


# #There are a few features (left of the plot) with higher mutual information values. There are also features with almost zero MI values on the right of the plot.
# 
# Once we find the mutual information values, to select features we need to determine a threshold, or cut-off value, above which a feature will be selected.
# 
# There are a few ways in which this can be done:
# 
# Select top k features, where k is an arbitrary number of features

# In[56]:


# select features
sel_ = SelectKBest(mutual_info_classif, k=10).fit(X_train, y_train)

# display features
X_train.columns[sel_.get_support()]


# In[60]:


class_count_0, class_count_1= df['Bankrupt'].value_counts()
print("class 0 count: ", class_count_0, "class 1 count:", class_count_1)
# Separate class
class_0 = df[df['Bankrupt'] == 0]
class_1 = df[df['Bankrupt'] == 1]

# print the shape of the class
print('class 0:', class_0.shape)
print('class 1:', class_1.shape)


# In[61]:


class_1.sample(class_count_0, replace = True).shape


# In[62]:


# Oversample 1-class and concat the DataFrames of both classes
class_1_over = class_1.sample(class_count_0, replace=True)
df_test_over = pd.concat([class_0, class_1_over], axis=0)


# In[63]:


print(df_test_over['Bankrupt'].value_counts())


# In[64]:


X_over = df_test_over.copy()
y_over = df_test_over['Bankrupt']
X_over = X_over.drop(['Bankrupt'], axis=1)


# In[65]:


X = df_test_over.drop('Bankrupt',axis='columns')
y = df_test_over['Bankrupt']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15, stratify=y)
X_train.shape, X_test.shape


# In[66]:


miseries = pd.Series(mi)
miseries
miseries.index = X_train.columns
miseries.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.ylabel('Mutual Information')


# In[67]:


# select features
k_best = SelectKBest(mutual_info_classif, k=10).fit(X_train, y_train)

# display features
X_train.columns[k_best.get_support()]


# In[68]:


# remove the rest of the features:
X_train = k_best.transform(X_train)
X_test = k_best.transform(X_test)
X_train.shape,X_test.shape


# In[69]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[73]:


lgclassifier = LogisticRegression(random_state = 42)
lgclassifier.fit(X_train, y_train)


# In[74]:


y_pred = lgclassifier.predict(X_test)
print(y_pred)


# In[75]:


acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}", "\nthe accuracy rate is lower after rebalancing using under sampling")


# In[76]:


c_matrix = confusion_matrix(y_test,y_pred)
#print confusion matrix
print(c_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,display_labels=["Not bankrupt", "Is bankrupt"])
disp.plot()
plt.show()


# In[77]:


#plots the ROC curves
plt.figure(figsize=(10, 6))
lg_probabilities = lgclassifier.predict_proba(X_test)[:, 1]

lg_auc = roc_auc_score(y_test, lg_probabilities)
lg_fpr, lg_tpr, lg_thresholds = roc_curve(y_test, lg_probabilities)
plt.plot(lg_fpr, lg_tpr, label=f"AUC - Logistic Classifier: {lg_auc}")

plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')

plt.xlabel('FPR (False Positive Rate)', size=14)
plt.ylabel('TPR (True Positive Rate)', size=14)
plt.title('Oversampling rebalanced - ROC Curve', size=18)
plt.legend()


# In[79]:


y_val_scores = lgclassifier.predict_proba(X_test)
print(y_val_scores)

y_train_scores = lgclassifier.predict_proba(X_train)
print(y_train_scores)


# In[81]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_val_scores[:,1])
print('AUC: ' + str(auc))


# In[82]:


predictions_val = lgclassifier.predict(X_test)


# In[83]:


from sklearn. metrics import classification_report

print('Validation Data Classification Report \n', classification_report(y_test, predictions_val))


# In[84]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[85]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[86]:


sns.heatmap(cm,annot=True)
plt.show()
acc3 = accuracy_score(y_test, y_pred)
print("Best Accuracy of K-NN: {:.2f} %".format(acc3*100))


# In[87]:


test_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred_test=knn_model.predict(X_test)
    test_error = 1-accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)
print(test_error_rates.index(min(test_error_rates)))


# In[88]:


print(classification_report(y_test,y_pred))


# In[89]:


plt.plot(range(1,30), test_error_rates)
plt.ylabel("Error Rate")
plt.xlabel("K Neighbors")


# In[90]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[91]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc6 = accuracy_score(y_test, y_pred)


# In[92]:


print(f"Random Forest Classification accuracy: {acc6}")


# In[93]:


c_matrix = confusion_matrix(y_test,y_pred)
#print confusion matrix
print(c_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,display_labels=["Not bankrupt", "Is bankrupt"])
disp.plot()
plt.show()


# In[94]:


print(classification_report(y_test,y_pred))


#  # Oversampling blindly copies the current samples to create new samples
#  

# # SMOTE:- Creates new samples from current samples usinng K nearest neighbours algorithm

# In[95]:


X = df.drop('Bankrupt',axis='columns')
y = df['Bankrupt']


# In[96]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
oversample = SMOTE()
X_sm, y_sm = smote.fit_resample(X, y)
y_sm.value_counts()


# # TRAIN and SPLIT

# In[97]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)


# In[98]:


# Number of classes in training Data
y_train.value_counts()


# # Feature selection

# In[99]:


miseries = pd.Series(mi)
miseries
miseries.index = X_train.columns
miseries.sort_values(ascending=False).plot.bar(figsize=(20, 6))
plt.ylabel('Mutual Information')


# In[100]:


# select features
k_best = SelectKBest(mutual_info_classif, k=10).fit(X_train, y_train)

# display features
X_train.columns[k_best.get_support()]


# In[101]:


# remove the rest of the features:
X_train = k_best.transform(X_train)
X_test = k_best.transform(X_test)
X_train.shape,X_test.shape


# # Feature Scaling

# In[102]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[103]:


lgclassifier = LogisticRegression(random_state = 42)
lgclassifier.fit(X_train, y_train)


# In[104]:


y_pred = lgclassifier.predict(X_test)
print(y_pred)


# In[105]:


acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}", "\nthe accuracy rate is increased after rebalancing using under sampling")


# In[106]:


c_matrix = confusion_matrix(y_test,y_pred)
#print confusion matrix
print(c_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,display_labels=["Not bankrupt", "Is bankrupt"])
disp.plot()
plt.show()


# In[107]:


#plots the ROC curves
plt.figure(figsize=(10, 6))
lg_probabilities = lgclassifier.predict_proba(X_test)[:, 1]

lg_auc = roc_auc_score(y_test, lg_probabilities)
lg_fpr, lg_tpr, lg_thresholds = roc_curve(y_test, lg_probabilities)
plt.plot(lg_fpr, lg_tpr, label=f"AUC - Logistic Classifier: {lg_auc}")

plt.plot([0, 1], [0, 1], color='blue', linestyle='--', label='Baseline')

plt.xlabel('FPR (False Positive Rate)', size=14)
plt.ylabel('TPR (True Positive Rate)', size=14)
plt.title('Oversampling rebalanced - ROC Curve', size=18)
plt.legend()


# In[108]:


from sklearn.metrics import roc_auc_score

auc = roc_auc_score(y_test,y_val_scores[:,1])
print('AUC: ' + str(auc))


# In[109]:


predictions_val = lgclassifier.predict(X_test)


# In[110]:


from sklearn. metrics import classification_report

print('Validation Data Classification Report \n', classification_report(y_test, predictions_val))


# # KNN

# In[111]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# In[112]:


y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[113]:


sns.heatmap(cm,annot=True)
plt.show()
acc3 = accuracy_score(y_test, y_pred)
print("Best Accuracy of K-NN: {:.2f} %".format(acc3*100))


# In[114]:


test_error_rates = []
for k in range(1,30):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    y_pred_test=knn_model.predict(X_test)
    test_error = 1-accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)
print(test_error_rates.index(min(test_error_rates)))


# In[115]:


print(classification_report(y_test,y_pred))


# In[116]:


plt.plot(range(1,30), test_error_rates)
plt.ylabel("Error Rate")
plt.xlabel("K Neighbors")


# # Random Forest Classifier

# In[117]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# In[118]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
acc6 = accuracy_score(y_test, y_pred)


# In[119]:


print(f"Random Forest Classification accuracy: {acc6}")


# In[120]:


c_matrix = confusion_matrix(y_test,y_pred)
#print confusion matrix
print(c_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix,display_labels=["Not bankrupt", "Is bankrupt"])
disp.plot()
plt.show()


# In[121]:


print(classification_report(y_test,y_pred))


# In[ ]:




