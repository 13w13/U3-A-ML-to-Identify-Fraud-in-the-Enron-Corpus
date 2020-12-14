#!/usr/bin/env python
# coding: utf-8

# # Machine Learning to Identify Fraud in the Enron Corpus

# In[1]:


import warnings 
warnings.filterwarnings("ignore")
import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
import sys
import pickle
import csv
import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
#from poi_data import *
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.metrics import accuracy_score, precision_score, recall_score


# ## Task 1: Select what features you'll use

# features_list is a list of strings, each of which is a feature name.    
# The first feature must be "poi".    
# features_list = ['poi','salary']    
# **You will need to use more features**

# In[2]:


target_label = 'poi'

email_features_list = [
    'from_messages',
    'from_poi_to_this_person',
    'from_this_person_to_poi',
    'shared_receipt_with_poi',
    'to_messages',
    ]
    
financial_features_list = [
    'bonus',
    'deferral_payments',
    'deferred_income',
    'director_fees',
    'exercised_stock_options',
    'expenses',
    'loan_advances',
    'long_term_incentive',
    'other',
    'restricted_stock',
    'restricted_stock_deferred',
    'salary',
    'total_payments',
    'total_stock_value',
]

features_list = [target_label] + financial_features_list + email_features_list


# In[3]:


### Load the dictionary containing the dataset

with open('final_project_dataset.pkl', 'rb') as data_file:
    data_dict = pickle.load(data_file)


# In[4]:


df = pd.DataFrame(data_dict)
df.T


# ### 1.1.0 Explore csv file 

# In[5]:


def make_csv(data_dict):
    """ generates a csv file from a data set"""
    fieldnames = ['name'] + data_dict.itervalues().next().keys()
    with open('data.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data_dict:
            person = data_dict[record]
            person['name'] = record
            assert set(person.keys()) == set(fieldnames)
            writer.writerow(person)


# ### 1.1.1 Dataset Exploration

# In[6]:


print('# Exploratory Data Analysis #')
data_dict.keys()
print('Total number of data points: %d' % len(data_dict.keys()))
num_poi = 0
for name in data_dict.keys():
    if data_dict[name]['poi'] == True:
        num_poi += 1
print('Number of Persons of Interest: %d' % num_poi)
print('Number of people without Person of Interest label: %d' % (len(data_dict.keys()) - num_poi))


# ### 1.1.2 Feature Exploration

# In[7]:


all_features = data_dict['ALLEN PHILLIP K'].keys()
print('Each person has %d features available' %  len(all_features))
### Evaluate dataset for completeness
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in all_features:
        if data_dict[person][feature] == 'NaN':
            missing_values[feature] += 1
        else:
            records += 1


# ### Print results of completeness analysis

# In[8]:


print('Number of Missing Values for Each Feature:')

#sorted(missing_values.values())

#for feature in all_features:
   # print("%s: %d" % (feature, sorted(missing_values.values())[feature])


for id in sorted(missing_values, key = missing_values.get, reverse = True):
          print(id, missing_values[id])


# => classification,we have here unblanced target. 
# Maybe Smot methodology ? 

# ## Task 2: Remove outliers

# In[9]:


def PlotOutlier(data_dict, feature_x, feature_y):
    """ Plot with flag = True in Red """
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()


# ### 2.1 Visualise outliers

# In[10]:


print(PlotOutlier(data_dict, 'total_payments', 'total_stock_value'))
print(PlotOutlier(data_dict, 'from_poi_to_this_person', 'from_this_person_to_poi'))
print(PlotOutlier(data_dict, 'salary', 'bonus'))
#Remove outlier TOTAL line in pickle file.
data_dict.pop( 'TOTAL', 0 )


# ### 2.2 Function to remove outliers

# In[11]:


def remove_outlier(dict_object, keys):
    """ removes list of outliers keys from dict object """
    for key in keys:
        dict_object.pop(key, 0)

outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
remove_outlier(data_dict, outliers)


# ### Task 3: Create new feature(s)

# ### 3.1 create new copies of dataset for grading

# In[12]:


my_dataset = data_dict


# ### 3.2 add new features to dataset

# In[13]:


def compute_fraction(x, y):
    """ return fraction of messages from/to that person to/from POI"""    
    if x == 'NaN' or y == 'NaN':
        return 0.
    if x == 0 or y == 0: 
        return 0
    fraction = x / y
    return fraction

def compute_ratio(poi_from, poi_to,messages_from, messages_to):
    """ return fraction of messages from/to that person to/from POI"""    
    if poi_from == 'NaN' or poi_to == 'NaN' or messages_from =='NaN' or messages_to=='NaN':
        return 0.
    fraction =  (poi_from + poi_to)/(messages_from + messages_to)
    return fraction


# By doing further research on the data and its source, we have learned that NaN values in financial and stock columns do not mean a lack of information but a zero value. So we will for each one of its columns replaced the NaN values by zeros.

# In[14]:


for name in my_dataset:
    data_point = my_dataset[name]
    NaN_value = 0
    if data_point['deferral_payments'] == 'NaN' :
        data_point['deferral_payments'] = NaN_value
        
    if data_point['total_payments'] == 'NaN' :
        data_point['total_payments'] = NaN_value
           
    if data_point['loan_advances'] == 'NaN':
        data_point['loan_advances'] = NaN_value
        
    if data_point['bonus'] == 'NaN' :
        data_point['bonus'] = NaN_value
        
    if data_point['restricted_stock_deferred'] == 'NaN':
        data_point['restricted_stock_deferred'] = NaN_value
    
    if data_point['total_stock_value'] == 'NaN' :
        data_point['total_stock_value'] = NaN_value
        
    if data_point['expenses'] == 'NaN' :
        data_point['expenses'] = NaN_value
        
    if data_point['exercised_stock_options'] == 'NaN' :
        data_point['exercised_stock_options'] = NaN_value   
        
    if data_point['long_term_incentive'] == 'NaN' :
        data_point['long_term_incentive'] = NaN_value
    
    if data_point['director_fees'] == 'NaN' :
        data_point['director_fees'] = NaN_value
    
    if data_point['director_fees'] == 'NaN' :
        data_point['director_fees'] = NaN_value


# Thanks to our research, we were able to identify FREVERT MARK A,LAVORATO JOHN J,WHALLEY LAWRENCE G and BAXTER JOHN C in the board of directors, nevertheless these 3 people are not POI. Therefore, we can anticipate that their very high financial data will distort our results in the future, and it is preferable to remove them from the dataset. 

# In[15]:


my_dataset.pop('FREVERT MARK A')
my_dataset.pop('LAVORATO JOHN J')
my_dataset.pop('WHALLEY LAWRENCE G')
my_dataset.pop('BAXTER JOHN C')


# In addition, we decided to replace the NaN values in the message columns with the average based on POI and non-POI employees. This will allow us to feed more information to our models. 

# In[16]:


cnt_from_poi_to_this_person =0
cnt_from_this_person_to_poi=0
cnt_to_messages =0
cnt_from_messages =0
cnt_shared_receipt_with_poi = 0

cnt_poi_from_poi_to_this_person =0
cnt_poi_from_this_person_to_poi=0
cnt_poi_to_messages =0
cnt_poi_from_messages =0
cnt_poi_shared_receipt_with_poi = 0

sum_poi_from_poi_to_this_person =0
sum_poi_from_this_person_to_poi=0
sum_poi_to_messages =0
sum_poi_from_messages =0
sum_shared_receipt_with_poi = 0

sum_from_poi_to_this_person =0
sum_from_this_person_to_poi=0
sum_to_messages =0
sum_from_messages =0
sum_poi_shared_receipt_with_poi = 0
    
for name in my_dataset:
    
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    from_messages = data_point['from_messages']
    to_messages = data_point['to_messages']
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    poi = data_point["poi"]
    shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
    
    if from_messages != 'NaN' and poi ==False:
        cnt_from_messages += 1
        sum_from_messages += from_messages
    elif from_messages != 'NaN' and poi ==True:
        cnt_poi_from_messages +=1
        sum_poi_from_messages += from_messages
    
    if to_messages != 'NaN' and poi ==False:
        cnt_to_messages += 1
        sum_to_messages += to_messages
    elif to_messages != 'NaN' and poi ==True:
        cnt_poi_to_messages +=1
        sum_poi_to_messages += to_messages
    
    if from_poi_to_this_person != 'NaN' and poi ==False:
        cnt_from_poi_to_this_person += 1
        sum_from_poi_to_this_person += from_poi_to_this_person
    elif from_messages != 'NaN' and poi ==True:
        cnt_poi_from_poi_to_this_person +=1
        sum_poi_from_poi_to_this_person+= from_poi_to_this_person
    
    if from_this_person_to_poi != 'NaN' and poi ==False:
        cnt_from_this_person_to_poi += 1
        sum_from_this_person_to_poi += from_this_person_to_poi 
    elif from_messages != 'NaN' and poi ==True:
        cnt_poi_from_this_person_to_poi +=1
        sum_poi_from_this_person_to_poi += from_this_person_to_poi 
        
    if shared_receipt_with_poi != 'NaN' and poi ==False:
        cnt_shared_receipt_with_poi += 1
        sum_shared_receipt_with_poi += shared_receipt_with_poi
    elif shared_receipt_with_poi != 'NaN' and poi ==True:
        cnt_poi_shared_receipt_with_poi +=1
        sum_poi_shared_receipt_with_poi += shared_receipt_with_poi
        
        
mean_from_poi_to_this_person = compute_fraction(sum_from_poi_to_this_person,cnt_from_poi_to_this_person)
mean_from_this_person_to_poi= compute_fraction(sum_from_this_person_to_poi, cnt_from_this_person_to_poi)
mean_to_messages =compute_fraction(sum_to_messages,cnt_to_messages)
mean_from_messages =compute_fraction(sum_from_messages,cnt_from_messages)
mean_shared_receipt_with_poi = compute_fraction(sum_shared_receipt_with_poi,cnt_shared_receipt_with_poi)

mean_poi_from_poi_to_this_person = compute_fraction(sum_poi_from_poi_to_this_person,cnt_poi_from_poi_to_this_person)
mean_poi_from_this_person_to_poi= compute_fraction(sum_poi_from_this_person_to_poi, cnt_poi_from_this_person_to_poi)
mean_poi_to_messages =compute_fraction(sum_poi_to_messages,cnt_poi_to_messages)
mean_poi_from_messages =compute_fraction(sum_poi_from_messages,cnt_poi_from_messages)
mean_poi_shared_receipt_with_poi = compute_fraction(sum_poi_shared_receipt_with_poi,cnt_poi_shared_receipt_with_poi)

for name in my_dataset:
    
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    from_messages = data_point['from_messages']
    to_messages = data_point['to_messages']
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
    poi = data_point["poi"]
    
    if from_messages == 'NaN' and poi ==False:
        data_point["from_messages"] = mean_from_messages
    elif from_messages == 'NaN' and poi ==True:
        data_point["from_messages"] = mean_poi_from_messages
        
    if to_messages == 'NaN' and poi ==False:
        data_point["to_messages"]== mean_to_messages
    elif to_messages == 'NaN' and poi ==True:
        data_point["to_messages"] = mean_poi_to_messages
    
    if from_poi_to_this_person == 'NaN' and poi ==False:
        data_point["from_poi_to_this_person"]  =mean_from_poi_to_this_person
    elif from_messages == 'NaN' and poi ==True:
        data_point["from_poi_to_this_person"]  = mean_poi_from_poi_to_this_person
    
    if from_this_person_to_poi == 'NaN' and poi ==False:
        data_point["from_this_person_to_poi"]  =mean_from_this_person_to_poi
    elif from_messages == 'NaN' and poi ==True:
        data_point["from_this_person_to_poi"]= mean_poi_from_this_person_to_poi
        
    if shared_receipt_with_poi == 'NaN' and poi ==False:
        data_point["shared_receipt_with_poi"]  =  mean_shared_receipt_with_poi
    elif from_messages == 'NaN' and poi ==True:
        data_point["shared_receipt_with_poi"]= mean_poi_shared_receipt_with_poi 
        


# In[17]:


print(mean_from_poi_to_this_person , mean_from_this_person_to_poi, mean_to_messages , mean_from_messages)
print(mean_poi_from_poi_to_this_person , mean_poi_from_this_person_to_poi , mean_poi_to_messages,mean_poi_from_messages )


# We add new ratio features :
#     
# 1. shared_recepeit with poi
# 2. bonus_to_salary
# 3. payments_to_salary
# 4. ratio mess
# 5. exercised_stock_options
# 6. bonus_to_total

# In[18]:


for name in my_dataset:
    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = compute_fraction(from_poi_to_this_person, to_messages)
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = compute_fraction(from_this_person_to_poi, from_messages)
    data_point["fraction_to_poi"] = fraction_to_poi
    
    shared_receipt_with_poi = data_point["shared_receipt_with_poi"]
    shared_receipt_poi_ratio = compute_fraction(shared_receipt_with_poi, to_messages)
    data_point["shared_receipt_poi_ratio"] = shared_receipt_poi_ratio
    
    bonus= data_point["bonus"]
    salary = data_point["salary"]
    bonus_to_salary = compute_fraction(bonus, salary)
    data_point["bonus_to_salary"] = bonus_to_salary  
    
    total_payments = data_point["total_payments"]
    bonus_to_total = compute_fraction(bonus, total_payments)
    data_point["bonus_to_total"] = bonus_to_total 
    
    exercised_stock_options= data_point["exercised_stock_options"]
    total_stock_value= data_point["total_stock_value"]
    exercised_stock_options_ratio = compute_fraction(exercised_stock_options, total_stock_value)
    data_point["exercised_stock_options_ratio"] = exercised_stock_options_ratio  
    
    ratio_mess= compute_ratio(from_poi_to_this_person, from_this_person_to_poi,from_messages, to_messages)
    data_point["ratio_mess"] = ratio_mess 


# Finally, while inquiring we found the members of the board and we wanted to add a feature where we indicate on a person is part of the board.

# In[19]:


for name in my_dataset:
    data_point = my_dataset[name]
    direction = 0 
    data_point["direction"] = direction 


# In[20]:


list_direction2 = ["LAY KENNETH L","SKILLING JEFFREY K"]
list_direction1 = ["BUY RICHARD B","CAUSEY RICHARD A","DERRICK JR. JAMES V","KEAN STEVEN J","KOENIG MARK E","METTS MARK","FASTOW ANDREW S","BAXTER JOHN C","HORTON STANLEY C","FREVERT MARK A","WHALLEY LAWRENCE G","PAI LOU L","WHITE JR THOMAS E","HIRKO JOSEPH","RICE KENNETH D"]
data_point = my_dataset[name]
for name in my_dataset : 
    for item in list_direction1 :
        if name == item : 
            direction = 1
            my_dataset[name]['direction'] = direction
    for item2 in list_direction2 :
        if name == item2 : 
            direction = 2
            my_dataset[name]['direction'] = direction


# ### 3.3 create new copies of feature list for grading

# In[21]:


my_feature_list = features_list +[ 'fraction_to_poi','shared_receipt_poi_ratio','bonus_to_salary','bonus_to_total','direction','ratio_mess','exercised_stock_options_ratio']


# In[22]:


features_list


# ### 3.4 get K-best features

# In[23]:


num_features = 10


# ### 3.5 function using SelectKBest

# In[24]:


def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    print(scores)
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print ("{0} best features: {1}\n".format(k, k_best_features.keys(), scores))
    return k_best_features


# => Maybe appropriate stat test for classification. 

# In[25]:


best_features = get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + list(set(best_features.keys()))


# ### 3.6 print features

# In[26]:


print ("{0} selected features: {1}\n".format(len(my_feature_list) - 1, my_feature_list[1:]))


# ### 3.7 extract the features specified in features_list
# 

# In[27]:


data = featureFormat(my_dataset, my_feature_list,sort_keys = True)


# split into labels and features

# In[28]:


labels, features = targetFeatureSplit(data)


# ### 3.8 scale features via min-max

# In[29]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


# ## Task 4: Using algorithm

# Please name your classifier clf for easy export below.   
# Note that if you want to do PCA or other multi-stage operations,    
# you'll need to use Pipelines. For more info:    
# http://scikit-learn.org/stable/modules/pipeline.html     
# 
# Provided to give you a starting point. Try a variety of classifiers.   

# ### 4.1  Gaussian Naive Bayes Classifier

# In[30]:


from sklearn.naive_bayes import GaussianNB
g_clf = GaussianNB()


# ### 4.2  Logistic Regression Classifier
# 

# In[31]:


from sklearn.linear_model import LogisticRegression


# In[32]:


l_clf = Pipeline(steps= [
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=1e-08, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, 
max_iter=100, multi_class='ovr', penalty='l2', random_state=42, solver='liblinear', tol=0.001, verbose=0))])


# ### 4.3  K-means Clustering

# In[33]:


from sklearn.cluster import KMeans
k_clf = KMeans(n_clusters=2, tol=0.001)


# ### 4.4 Support Vector Machine Classifier

# In[34]:


from sklearn.svm import SVC
s_clf = SVC(kernel='rbf', C=1000,gamma = 0.0001,random_state = 42, class_weight = 'balanced')


# ### 4.5 Random Forest
# 

# In[35]:


from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(max_depth = 5,max_features = 'sqrt',n_estimators = 10, random_state = 42)


# ### 4.6 Gradient Boosting Classifier

# In[36]:


from sklearn.ensemble  import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=10,random_state = 42)


# ### 4.7 Decision Tree Classifier

# In[37]:


from sklearn.tree import DecisionTreeClassifier


# In[38]:


tre_clf=DecisionTreeClassifier(random_state=42)


# ### 4.8 KNeighborsClassifier

# In[39]:


from sklearn.neighbors import KNeighborsClassifier


# In[40]:


knn_clf = KNeighborsClassifier(n_neighbors=3)


# ### 4.9 Perceptron

# In[41]:


from sklearn.linear_model import Perceptron
pe_clf= Perceptron(max_iter=5)


# ### 4.10 MLP Perceptron

# In[42]:


from sklearn.neural_network import MLPClassifier


# In[43]:


mlp_clf = MLPClassifier(random_state=1)


# ### 4.9 evaluate function
# 

# In[44]:


from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)


# In[45]:


def evaluate_clf(clf, features, labels, num_iters=1000, test_size=0.3):
    print (clf)
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =            train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train,labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False

    print ("done.\n")
    print ("precision: {}".format(mean(precision)))
    print ("recall:    {}".format(mean(recall)))
    print ("accuracy:    {}".format(mean(accuracy)))
    return len(labels_test)
    return mean(precision), mean(recall)


# ### 4.8 Evaluate all functions

# In[46]:


evaluate_clf(g_clf, features, labels)
evaluate_clf(l_clf, features, labels)
evaluate_clf(k_clf, features, labels)
evaluate_clf(s_clf, features, labels)
evaluate_clf(rf_clf, features, labels)
evaluate_clf(gb_clf, features, labels)
evaluate_clf(tre_clf, features, labels)
evaluate_clf(knn_clf, features, labels)
evaluate_clf(pe_clf, features, labels)
evaluate_clf(mlp_clf, features, labels)


# ### 5. Hyperparameters tuning

# In[47]:


from sklearn.model_selection import GridSearchCV
import numpy as np


# #### 5.1 Decision tree
We select one of the best model from the evaluate function, in order to tune the hyperparameters. 
As a reminder, here are the results obtained without tuning :
    precision: 0.4703893550893551
    recall:    0.45136847041847045
    accuracy:    0.858238095238095

First, we create a pipeline to run a GridSearch on the select K best as to find the best features selection number.
# In[48]:


n_features = np.arange(1, 20)
my_feature_list = features_list +['fraction_to_poi','shared_receipt_poi_ratio','bonus_to_salary','bonus_to_total','direction','ratio_mess','exercised_stock_options_ratio']
data = featureFormat(my_dataset, my_feature_list,sort_keys = True)
labels, features = targetFeatureSplit(data)
# Create a pipeline with feature selection and classification
pipe_k1 = Pipeline([
    ('select_features', SelectKBest()),
    ('classifier',DecisionTreeClassifier())])
param_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=67)
k_clf= GridSearchCV(pipe_k1, param_grid=param_grid, scoring='f1', cv = cv)
k_clf.fit(features, labels)


# In[49]:


k_clf.best_score_


# In[50]:


k_clf.best_score_
k_clf.best_estimator_


# In[51]:


num_features=19


# In[52]:


best_features = get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + list(set(best_features.keys()))


# In[53]:


data = featureFormat(my_dataset, my_feature_list,sort_keys = True)
labels, features = targetFeatureSplit(data)

The best number of features is 19.
Now we are interested in the internal parameters of the decision tree.
Then, we launch a new search grid where we test differents parameters as : 
- criterion
- max depth
- min samples split
- min samples leaf 
The grid search is scored based on the f1 result in order to optimize the recall and the precision.
# In[54]:


clf_parameters = { 'criterion': ['gini', 'entropy'],
                   'max_depth': [None, 1, 2, 4, 5, 10, 15, 20],
                   'min_samples_split': [2, 4, 6, 8, 10, 20, 30, 40],
                   'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 10, 20, 30] }

cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=67)
clf = GridSearchCV(DecisionTreeClassifier(), param_grid = clf_parameters, cv = cv, scoring = 'f1')
clf.fit(features,labels)

clf.best_estimator_


# In[55]:


clf.best_params_

The best parameters are : {'criterion': 'gini',
 'max_depth': None,
 'min_samples_leaf': 1,
 'min_samples_split': 20}
 We therefore launch the evaluation of our model.
# In[94]:


clf_best_tree=DecisionTreeClassifier(criterion= 'gini',
 max_depth = None,
 min_samples_leaf = 1,
 min_samples_split = 20)


# In[95]:


evaluate_clf(clf_best_tree,features,labels)

With this evaluation, we have a result of : 
    precision: 0.4970781787656788
    recall:    0.6162624458874459
    accuracy:    0.8779761904761906
Now we can test with the tester made by our excellent teacher.
# In[58]:


import tester
tester.dump_classifier_and_data(clf_best_tree , my_dataset, my_feature_list)
tester.main()

With the tester, we gain in precision however at the same time we lost in recall. 
Our result :
Accuracy: 0.88864	Precision: 0.67971	Recall: 0.41700 F1: 0.51689	F2: 0.45193
# #### 5.2 Log Regression
We have identified that logistic regression is the one of the most efficient model with our dataset. We want to do hyperparameter tuning in order to improve our results. First of all, we want to analyze the best number of parameters to feed our model. Therefore we create a pipeline with a selectKbest with as parameters the numbers of features.
# In[59]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV


# In[60]:


n_features = np.arange(1, 20)
my_feature_list = features_list +['fraction_to_poi','shared_receipt_poi_ratio','bonus_to_salary','bonus_to_total','direction','ratio_mess','exercised_stock_options_ratio']
data = featureFormat(my_dataset, my_feature_list,sort_keys = True)
labels, features = targetFeatureSplit(data)
# Create a pipeline with feature selection and classification
pipe_k = Pipeline([
    ('scaler', StandardScaler()),
    ('select_features', SelectKBest()),
    ('classifier', LogisticRegression())])
param_grid = [
    {
        'select_features__k': n_features
    }
]

# Use GridSearchCV to automate the process of finding the optimal number of features
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=67)
k_lcf= GridSearchCV(pipe_k, param_grid=param_grid, scoring='f1', cv = cv)
k_lcf.fit(features, labels)

# Use GridSearchCV to automate the process of finding the optimal number of features

cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=67)
k_clf= GridSearchCV(pipe_k1, param_grid=param_grid, scoring='f1', cv = 10)
k_clf.fit(features, labels)


# In[61]:


k_lcf.best_score_
k_lcf.best_estimator_

As we can see, the best results appear with 7 features.
Now we are interested in the internal parameters of the logistic regression.
# In[62]:


num_features=7


# In[63]:


best_features = get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + list(set(best_features.keys()))


# In[64]:


data = featureFormat(my_dataset, my_feature_list,sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[65]:


pipe_log = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())])

Now we are interested in the internal parameters of the logistic regression.
Then, we launch a new search grid where we test differents parameters as : 
- solvers
- penalty
- c_values
- class weight
- multi class
The grid search is scored based on the f1 result in order to optimize the recall and the precision.
# In[66]:


# define models and parameters

solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ["l1","l2","elasticnet","none"]
c_values = np.logspace(-4, 4, 50)
class_weight=['balanced',None]
multi_class=["ovr"]

# define grid search
grid = dict(classifier__solver=solvers,classifier__penalty=penalty,classifier__C=c_values,classifier__class_weight=class_weight,classifier__multi_class=multi_class)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, train_size=0.7,random_state=1)
grid_search = GridSearchCV(estimator=pipe_log, param_grid=grid, n_jobs=-1, cv=cv,scoring = 'f1')
grid_result = grid_search.fit(features, labels)


# In[67]:


grid_result.best_estimator_


# In[68]:


grid_result.best_params_

The best parameters are : {'C': 0.009102981779915217,
 'class weight': None,
 'multi_class': 'ovr',
 'penalty': 12,
 'solver': liblinear }
 We therefore launch the evaluation of our model.
# In[69]:


clf_best_log_f1=Pipeline(steps=[('std_slc', StandardScaler()),
                ('logistic_Reg',
                 LogisticRegression(C=0.009102981779915217,
                                    class_weight=None, multi_class='ovr',penalty= 'l2',
                                    solver='liblinear', tol=0.001))])


# Then we run the evaluate with our tunes parameters 

# In[70]:


evaluate_clf(clf_best_log_f1,features,labels)

With this evaluation, we have a result of : 
    precision: 0.5337182012432011
    recall:    0.6074877344877344
    accuracy:    0.8709285714285716
Now we can test with the tester made by our excellent teacher.
# In[71]:


import tester
tester.dump_classifier_and_data(clf_best_log_f1 , my_dataset, my_feature_list)
tester.main()


# With the tester, we gain in precision however at the same time we lost in recall. 
# Our result :
# Accuracy: 0.88736	Precision: 0.60302	Recall: 0.61900	F1: 0.61091	F2: 0.6157

# #### Percetron

# In[72]:


n_features = np.arange(1, 20)
my_feature_list = features_list +['fraction_to_poi','shared_receipt_poi_ratio','bonus_to_salary','bonus_to_total','direction','ratio_mess','exercised_stock_options_ratio']
data = featureFormat(my_dataset, my_feature_list,sort_keys = True)
labels, features = targetFeatureSplit(data)
# Create a pipeline with feature selection and classification
pipe_p = Pipeline([
    ('scaler', preprocessing.MinMaxScaler()),
    ('select_features', SelectKBest()),
    ('classifier', Perceptron())])
param_grid = [
    {
        'select_features__k': n_features
    }]

# Use GridSearchCV to automate the process of finding the optimal number of features
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=67)
k_lcf= GridSearchCV(pipe_p, param_grid=param_grid, scoring='f1', cv = cv)
k_lcf.fit(features, labels)


# In[73]:


k_lcf.best_score_
k_lcf.best_params_

As we can see, the best results appear with 11 features.
Now we are interested in the internal parameters of the logistic regression.
# In[74]:


num_features=11


# In[75]:


best_features = get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + list(set(best_features.keys()))


# In[76]:


data = featureFormat(my_dataset, my_feature_list,sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[77]:


pipe_per = Pipeline([
    ('scaler', preprocessing.MinMaxScaler()),
    ('classifier', Perceptron())])


# In[78]:


# define models and parameters

penalty = ["l1","l2","elasticnet","none"]
alpha = np.logspace(-4, 4, 50)
fit_intercept = [True, False]
shuffle = [True, False]
class_weight=['balanced',None]

# define grid search
grid = dict(classifier__penalty=penalty,classifier__alpha=alpha,classifier__class_weight=class_weight,classifier__shuffle=shuffle,classifier__fit_intercept=fit_intercept)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.3, train_size=0.7,random_state=1)
grid_search = GridSearchCV(estimator=pipe_per, param_grid=grid, n_jobs=-1, cv=cv,scoring = 'f1')
grid_result = grid_search.fit(features, labels)


# In[79]:


grid_result.best_estimator_


# In[80]:


grid_result.best_params_


# In[81]:


clf_best_per_f1=Pipeline(steps=[('scaler', MinMaxScaler()),
                ('classifier',
                 Perceptron(alpha=0.0020235896477251557, penalty='l1',
                            shuffle=False))])


# In[82]:


evaluate_clf(clf_best_per_f1,features,labels)


# In[83]:


import tester
tester.dump_classifier_and_data(clf_best_per_f1, my_dataset, my_feature_list)
tester.main()


# Best model after parameters tunning :    
#     Logistic regression with 11 features. 

# #### 5.4 Try staking the 2 models

# In[84]:


#pip install mlxtend  


# In[85]:


from mlxtend.classifier import StackingClassifier


# In[86]:


num_features=7


# In[87]:


best_features = get_k_best(my_dataset, my_feature_list, num_features)
my_feature_list = [target_label] + list(set(best_features.keys()))


# In[88]:


m_clf = StackingClassifier(classifiers=[clf_best_log_f1,clf_best_tree,clf_best_per_f1],use_probas=False,meta_classifier=clf_best_log_f1)


# In[89]:


evaluate_clf(m_clf,features,labels)


# In[90]:


import tester
tester.dump_classifier_and_data(m_clf, my_dataset, my_feature_list)
tester.main()


# #### Our model selection : logistic regression 

# Select Logistic Regression as final algorithm

# When we compare the result from the evaluate function and the tester, it seems that our logistic regression model has the best score after paramaters tuning. This result confirms the relevance of our pre-processing step. We don't think that we can increase anymore our recall and precision without more information about the dataset.

# In[91]:


clf = clf_best_log_f1


# dump your classifier, dataset and features_list so   
# anyone can run/check your results

# In[92]:


pickle.dump(clf, open("../final_project/my_classifier.pkl", "wb"))
pickle.dump(my_dataset, open("../final_project/my_dataset.pkl", "wb"))
pickle.dump(my_feature_list, open("../final_project/my_feature_list.pkl", "wb"))


# ### Task 6: Dump your classifier, dataset, and features_list

# Task 6: Dump your classifier, dataset, and features_list so anyone can   
# check your results. You do not need to change anything below, but make sure  
# that the version of poi_id.py that you submit can be run on its own and   
# generates the necessary .pkl files for validating your results.  

# In[93]:


dump_classifier_and_data(clf, my_dataset, features_list)


# In[10]:


jupyter nbconvert --to script *.ipynb

