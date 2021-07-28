#!/usr/bin/env python
# coding: utf-8

# # <div style="text-align: center">Gmail Classification Models</div>

# **Import Libraries**

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


# **Read Excel file**

# In[2]:


df = pd.read_excel(r'C:\Users\mgman\Downloads\All_Emails.xlsx')

df.drop('Unnamed: 0', axis=1, inplace = True)
df.columns = ['Label', 'Text', 'Label_Number']
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.isna().sum()


# In[6]:


df['Label_Number'].value_counts()


# **Count Plot**

# In[7]:


plt.figure(figsize = (8, 6))
sns.countplot(data = df, x = 'Label');


# **Count no. of each word**

# In[8]:


def count_words(text):
    words = word_tokenize(text)
    return len(words)
df['count']=df['Text'].apply(count_words)
df['count']


# In[9]:


df.groupby('Label_Number')['count'].mean()


# **Tokenization**

# In[10]:


get_ipython().run_cell_magic('time', '', 'def clean_str(string, reg = RegexpTokenizer(r\'[a-z]+\')):\n    # Clean a string with RegexpTokenizer\n    string = string.lower()\n    tokens = reg.tokenize(string)\n    return " ".join(tokens)\n\nprint(\'Before cleaning:\')\ndf.head()')


# In[11]:


print('After cleaning:')
df['Text'] = df['Text'].apply(lambda string: clean_str(string))
df.head()


# **Stemming words**

# In[12]:


from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stemming (text):
    return ''.join([stemmer.stem(word) for word in text])
df['Text']=df['Text'].apply(stemming)
df.head()


# In[13]:


X = df.loc[:, 'Text']
y = df.loc[:, 'Label_Number']

print(f"Shape of X: {X.shape}\nshape of y: {y.shape}")


# **Split into Training data and Test data**

# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)


# In[15]:


print(f"Training Data Shape: {X_train.shape}\nTest Data Shape: {X_test.shape}")


# **Count Vectorization to Extract Features from Text**

# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
cv.fit(X_train)


# In[17]:


print('No.of Tokens: ',len(cv.vocabulary_.keys()))


# In[18]:


dtv = cv.transform(X_train)
type(dtv)


# In[19]:


dtv = dtv.toarray()


# In[20]:


print(f"Number of Observations: {dtv.shape[0]}\nTokens/Features: {dtv.shape[1]}")


# In[21]:


dtv[1]


# **Apply different models**

# In[22]:


get_ipython().run_cell_magic('time', '', 'from sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.naive_bayes import MultinomialNB\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.svm import LinearSVC, SVC\nfrom time import perf_counter\nimport warnings\nwarnings.filterwarnings(action=\'ignore\')\nmodels = {\n    "Random Forest": {"model":RandomForestClassifier(), "perf":0},\n    "MultinomialNB": {"model":MultinomialNB(), "perf":0},\n    "Logistic Regr.": {"model":LogisticRegression(solver=\'liblinear\', penalty =\'l2\' , C = 1.0), "perf":0},\n    "KNN": {"model":KNeighborsClassifier(), "perf":0},\n    "Decision Tree": {"model":DecisionTreeClassifier(), "perf":0},\n    "SVM (Linear)": {"model":LinearSVC(), "perf":0},\n    "SVM (RBF)": {"model":SVC(), "perf":0}\n}\n\nfor name, model in models.items():\n    start = perf_counter()\n    model[\'model\'].fit(dtv, y_train)\n    duration = perf_counter() - start\n    duration = round(duration,2)\n    model["perf"] = duration\n    print(f"{name:20} trained in {duration} sec")')


# In[23]:


test_dtv = cv.transform(X_test)
test_dtv = test_dtv.toarray()
print(f"Number of Observations: {test_dtv.shape[0]}\nTokens: {test_dtv.shape[1]}")


# **Test Accuracy and Training Time**

# In[24]:


models_accuracy = []
for name, model in models.items():
    models_accuracy.append([name, model["model"].score(test_dtv, y_test),model["perf"]])


# In[25]:


df_accuracy = pd.DataFrame(models_accuracy)
df_accuracy.columns = ['Model', 'Test Accuracy', 'Training time (sec)']
df_accuracy.sort_values(by = 'Test Accuracy', ascending = False, inplace=True)
df_accuracy.reset_index(drop = True, inplace=True)
df_accuracy


# In[26]:


plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y ='Test Accuracy', data = df_accuracy)
plt.title('Accuracy on the test set\n', fontsize = 15)
plt.ylim(0.825,1)
plt.show()


# In[56]:


plt.figure(figsize = (15,5))
sns.barplot(x = 'Model', y = 'Training time (sec)', data = df_accuracy)
plt.title('Training time for each model in sec', fontsize = 15)
plt.ylim(0,1)
plt.show()


# ## **Logistic Regression**<br>

# In[28]:


get_ipython().run_cell_magic('time', '', "lr = LogisticRegression(solver='liblinear', penalty ='l2' , C = 1.0)\nlr.fit(dtv, y_train)\npred = lr.predict(test_dtv)")


# In[29]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# **Classification Report**

# In[30]:


print(classification_report(y_test, pred))


# **Confusion Matrix**

# In[31]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ## **Support Vector Machine (RBF)**<br>

# In[32]:


get_ipython().run_cell_magic('time', '', 'svc = SVC()\nsvc.fit(dtv, y_train)\npred = svc.predict(test_dtv)')


# In[33]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# **Classification Report**

# In[34]:


print(classification_report(y_test, pred))


# **Confusion Matrix**

# In[35]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ## **Random Forest Classifier**<br>

# In[36]:


get_ipython().run_cell_magic('time', '', 'rfc = RandomForestClassifier()\nrfc.fit(dtv, y_train)\npred = rfc.predict(test_dtv)')


# In[37]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# **Classification Report**

# In[38]:


print(classification_report(y_test, pred))


# **Confusion Matrix**

# In[39]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ## **Multinomial Naive Bayes** <br>

# In[40]:


get_ipython().run_cell_magic('time', '', 'mnb = MultinomialNB()\nmnb.fit(dtv, y_train)\npred = mnb.predict(test_dtv)')


# In[41]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# **Classification Report**

# In[42]:


print(classification_report(y_test, pred))


# **Confusion Matrix**

# In[43]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ## **Support Vector Machine (Linear)** <br>

# In[44]:


get_ipython().run_cell_magic('time', '', 'lsvc = LinearSVC()\nlsvc.fit(dtv, y_train)\npred = lsvc.predict(test_dtv)')


# In[45]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# **Classification Report**

# In[46]:


print(classification_report(y_test, pred))


# **Confusion Matrix**

# In[47]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ## **Decision Tree Classifier** <br>

# In[76]:


get_ipython().run_cell_magic('time', '', 'dtc = DecisionTreeClassifier()\ndtc.fit(dtv, y_train)\npred = dtc.predict(test_dtv)')


# In[77]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# **Classification Report**

# In[78]:


print(classification_report(y_test, pred))


# **Confusion Matrix**

# In[79]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);


# ## **K Nearest Neighbours**<br>

# In[52]:


get_ipython().run_cell_magic('time', '', 'knn = KNeighborsClassifier()\nknn.fit(dtv, y_train)\npred = knn.predict(test_dtv)')


# In[53]:


print('Accuracy: ', accuracy_score(y_test, pred) * 100)


# **Classification Report**

# In[54]:


print(classification_report(y_test, pred))


# **Confusion Matrix**

# In[55]:


confusion_matrix = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
plt.figure(figsize = (6, 6))
sns.heatmap(confusion_matrix, annot = True, cmap = 'Paired', cbar = False, fmt="d", xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam']);

