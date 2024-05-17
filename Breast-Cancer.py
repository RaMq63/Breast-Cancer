#!/usr/bin/env python
# coding: utf-8

# In[62]:


import sklearn as sk
import pandas as pd
import matplotlib as mp


# In[63]:


from sklearn.datasets import load_breast_cancer


# In[64]:


data = load_breast_cancer()


# In[65]:


label_names = data['target_names']
labels =data['target']
feature_names = data['feature_names']
features = data['data']


# In[66]:


print(label_names)
print(labels[0])
#the malignant means no cancer and equals 0, the bengign means a cancer and equals 1


# In[67]:


print(feature_names)
print()
print(features[0])


# In[68]:


df = pd.DataFrame(data = data.data, columns = data.feature_names)
df.head()


# In[69]:


from sklearn.model_selection import train_test_split


# In[70]:


#spliting the dataset for training and testing
train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.40, random_state = 42)


# In[71]:


#Build the Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[72]:


#Initilize the decision tree classifier 
dtc = DecisionTreeClassifier()

#train the classifier on the training set
dtc.fit(train, train_labels)


# In[73]:


#make prediction on the test data
perd = dtc.predict(test)


# In[74]:


#calcualt the accuracy
accuracy = accuracy_score(test_labels, perd)
print("The Accuracy=", accuracy)


# In[75]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


# In[85]:


# Function to plot precision-recall curve
def plot_precision_recall_curve(true, y_scores):
    precision, recall, _ = precision_recall_curve(true, y_scores)
    plt.figure(figsize=(3, 3))
    plt.plot(recall, precision, color='b', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Breast Cancer')
    plt.legend()
    plt.show()
    
def plot_confusion_matrix(true, pred, labels):
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predected Labels')
    plt.ylabel('True Label')
    plt.title('Breast Cancer')
    plt.show()
    
    
def roc_curve(fpr, tpr):
    plt.figure(figsize=(3, 3))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('benign')
    plt.ylabel('malignant')
    plt.title('Breast Cancer')
    plt.legend(loc="lower right")
    plt.show()

# Sample true labels and predicted scores
true = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
y_scores = [0.1, 0.9, 0.3, 0.8, 0.2, 0.7, 0.4, 0.6, 0.5, 0.75]

# Example visualizations
plot_confusion_matrix(true, [1 if score > 0.5 else 0 for score in y_scores], labels=['Negative', 'Positive'])
plot_precision_recall_curve(true, y_scores)
roc_curve(test_labels,dtc.predict_proba(test)[:,1])


# In[82]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plot_tree(dtc, filled=True, feature_names=df.columns, class_names=['Benign', 'Malignant'])
plt.title('Decision Tree Visualization')
plt.show()

