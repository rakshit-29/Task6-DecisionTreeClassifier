#!/usr/bin/env python
# coding: utf-8

# # Prediction using Decision Tree Classifier algorithm on Iris Dataset | TASK 6
# 

# ## Data Science & Business Analytics

# ## The Sparks Foundation GRIP MAY 2021
# 

# ## Task 6-
# 
# #### From the 'Iris' dataset with species name, create a decision tree classifier and represent it visually. The purpose is if we feed any new data to this classifier, it would be able to predict the right class accordingly.

# ## Author: Rakshit Saxena

# In[1]:


##Importing all the useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Reading the Iris.csv file 

Iris_data = pd.read_csv('E:\Iris.csv')


# In[3]:


#Checking top 10 records of Dataset..
Iris_data.head(10)


# In[4]:


#Gives us the information about our dataset which is a pandas dataframe. There are 6 columns and their type is mentioned. 
Iris_data.info()


# *Iris_data contains total 6 features in which 4 features(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalwidthCm) are independent features and 1 feature(Species) is dependent or target variable*

# In[5]:


#Describe function gives the basic numerical info about data for each numeric feature.

Iris_data.describe()


# In[6]:


#Data points count value for each class labels

Iris_data.Species.value_counts()


# **All Independent features have not-null float values and target variable has class labels(Iris-setosa, Iris-versicolor, Iris-virginica)**

# ## Visualization of the Iris.csv data

# In[7]:


#Sepal length vs sepal width
plt.scatter(x=Iris_data['SepalLengthCm'],y=Iris_data['SepalWidthCm'], color='lawngreen')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.show()


# In[8]:


#Petal length vs Petal width
plt.scatter(x=Iris_data['PetalLengthCm'],y=Iris_data['PetalWidthCm'], color='mediumvioletred')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()


# In[9]:


#Using Seaborn to visualise sepal length vs sepal width on target variable.
sns.set_style('darkgrid')
sns.FacetGrid(Iris_data,palette="rocket_r", hue = 'Species')    .map(plt.scatter, 'SepalLengthCm','SepalWidthCm')    .add_legend()

plt.show()


# In[10]:


#Using Seaborn to visualise petal length vs petal width based on target variable.
sns.set_style('darkgrid')
sns.FacetGrid(Iris_data,palette="winter_r", hue = 'Species')    .map(plt.scatter, 'PetalLengthCm','PetalWidthCm')    .add_legend()

plt.show()


# In[11]:


#Pair plot for our iris data
sns.pairplot(Iris_data.drop(['Id'],axis=1), hue='Species', palette="rocket_r")
plt.show()


# # **Adding features to our dataset**

# ##### **Adding 2 columns, Sepal-petal length difference and Sepal-petal width difference**

# In[12]:


Iris_data['Sepal_petal_len_diff'] = Iris_data['SepalLengthCm']-Iris_data['PetalLengthCm']
Iris_data['Sepal_petal_width_diff'] = Iris_data['SepalWidthCm']-Iris_data['PetalWidthCm']
Iris_data


# In[13]:


#Sepal_petal_len_diff vs Sepal_petal_width_diff
sns.set_style('darkgrid')
sns.FacetGrid(Iris_data,hue='Species', palette='rocket_r')   .map(plt.scatter,'Sepal_petal_len_diff','Sepal_petal_width_diff')   .add_legend()
plt.show()

sns.set_style('darkgrid')
sns.FacetGrid(Iris_data,hue='Species', palette='rocket_r')   .map(sns.distplot,'PetalLengthCm')   .add_legend()
plt.show()


# ##### **Adding 2 more columns, Sepal length and width difference and Petal length and width difference**

# In[14]:


Iris_data['Sepal_diff'] = Iris_data['SepalLengthCm']-Iris_data['SepalWidthCm']
Iris_data['Petal_diff'] = Iris_data['PetalLengthCm']-Iris_data['PetalWidthCm']
Iris_data


# In[15]:


#Sepal_diff vs Petal_diff
sns.set_style('darkgrid')
sns.FacetGrid(Iris_data,hue='Species', palette="rocket_r")   .map(plt.scatter,'Sepal_diff','Petal_diff')   .add_legend()
plt.show()    


sns.set_style('darkgrid')
sns.FacetGrid(Iris_data,hue='Species', palette="rocket_r")   .map(sns.distplot,'Petal_diff')   .add_legend()
plt.show()


# In[16]:


# All 4 new features
sns.pairplot(Iris_data[['Species', 'Sepal_diff', 'Petal_diff', 'Sepal_petal_len_diff',       'Sepal_petal_width_diff']], hue='Species', palette="rocket_r")
plt.show()


# In[17]:


#Droping ID column because it is uneccessary for our training of the machine learning model

Iris_data.drop(['Id'],axis=1,inplace=True)


# In[18]:


Iris_data #As we can see the ID column is not there anymore


# In[19]:


#Exploring distribution plot for all features

for i in Iris_data.columns:
    if i == 'Species':
        continue
    sns.set_style('dark')
    sns.FacetGrid(Iris_data,hue='Species', palette='magma')    .map(sns.distplot,i)    .add_legend()
    plt.show()


# ## Decision Tree Classifier model building

# In[20]:


#As per our analysis, we can't find much information from new features
#We will use the new features for classification

'''Imporing libraries to create Decision tree classifier and visualize the tree structure'''

from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score


'''Separating independent varibles/target varibles from the dataset'''


X = Iris_data[['SepalLengthCm', 'SepalWidthCm','PetalLengthCm', 'PetalWidthCm',               'Sepal_petal_len_diff','Sepal_petal_width_diff','Sepal_diff','Petal_diff',]]
y = Iris_data['Species']


#Before training the model we have to split the data into an Actual Train and Actual Test Dataset
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.30, random_state=42)

#Spliting data into validation train and validation test
Xt, Xcv, Yt, Ycv = train_test_split(Xtrain, Ytrain, test_size=0.10, random_state=42)


'''Now we have create a Decision tree classifier and trained it with training dataset.'''


Iris_clf = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
Iris_clf.fit(Xt, Yt)

#Visualizing the Tree which is formed on trained dataset
tree.plot_tree(Iris_clf)


# In[21]:


#Visualizing Decision Tree using graphviz library
dot_data = tree.export_graphviz(Iris_clf, out_file=None)

graph = graphviz.Source(dot_data)
graph


# In[31]:


#Decision tree using cross validation method to get the accuracy or performance score of our model.
print('Accuracy score is:',cross_val_score(Iris_clf, Xt, Yt, cv=3, scoring='accuracy').mean()*100,'%')


# In[32]:


#Checking validation test data on our trained model and getting performance metrices

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

Y_hat = Iris_clf.predict(Xcv)


print('Accuracy score for validation test data is:',accuracy_score(Ycv, Y_hat)*100,'%')
multilabel_confusion_matrix(Ycv , Y_hat)


# In[24]:


#Checking our model performance on actual unseen test data 
YT_hat = Iris_clf.predict(Xtest)
YT_hat

print('Model Accuracy Score on totally unseen data(Xtest) is:',accuracy_score(Ytest, YT_hat)*100,'%')
multilabel_confusion_matrix(Ytest , YT_hat)


# In[25]:


'''Training model on Actual train data... '''
Iris_Fclf = DecisionTreeClassifier(criterion='gini',min_samples_split=2)
Iris_Fclf.fit(Xtrain, Ytrain)

#Visualizing tree structure..
tree.plot_tree(Iris_Fclf)


# In[26]:


#Final Decision tree build for deployment

dot_data = tree.export_graphviz(Iris_Fclf, out_file=None)
graph = graphviz.Source(dot_data)
graph


# In[27]:


#Checking the performance of model on Actual Test data...

YT_Fhat = Iris_Fclf.predict(Xtest)
YT_Fhat

print('Model Accuracy Score on totally unseen data(Xtest) is:',accuracy_score(Ytest, YT_Fhat)*100,'%')
multilabel_confusion_matrix(Ytest , YT_Fhat)


# In[28]:


#Predicting the new points, except from Dataset

Test_point = [[5.1, 3.6, 1.4, 0.2, 3.7, 3.3, 1.6, 1.2],
             [6.9, 3.2, 5.2, 2.5, 1.4, 1.1, 3.9, 2.8],
             [6.2, 3.6, 2.4, 1.6, 3.9, 3.3, 1.9, 1.9],
             [5.5, 4.5, 5.5, 0.2, 4.0, 3.3, 1.6, 1.1],
             [7.5, 3.1, 2.3, 2.9, 1.4, 1.4, 3.9, 2.2],
             [6.5, 3.7, 1.1, 2.8, 4.0, 1.1, 4.6, 2.9]]

print(Iris_Fclf.predict(Test_point))


# # Thank you.

# In[ ]:




