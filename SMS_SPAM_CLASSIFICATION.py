#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the required libraries
import numpy as np
import pandas as pd


# In[2]:


#import data set
dataset = pd.read_csv("smsspam",sep='\t',names=['label','message'])
#sep because the dataset data ,is not seperated by comma whereas its seperated by tabs 
dataset


# In[3]:


dataset.isnull()


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


dataset = dataset.fillna(method='bfill')
dataset


# In[7]:


dataset["label"] = dataset["label"].map({"ham":0,"spam":1})


# In[8]:


dataset


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


plt.figure(figsize=(8,8))
g = sns.countplot(x="label",data=dataset)
plt.title("Countplot for sam vs ham ")
plt.xlabel('Is the SMS spam?')
plt.ylabel('COunt')


# In[11]:


#Handling the imbalamced data set usng Oversampling
only_spam= dataset[dataset['label']==1]
only_spam


# In[12]:


#no.of spam sms
print(len(only_spam))
#no.of ham sms
print(len(dataset)-len(only_spam))


# In[13]:


count = int((dataset.shape[0]-only_spam.shape[0])/only_spam.shape[0])
count


# In[14]:


#to balance the dataset
for i in range (0,count-1):
    dataset=pd.concat([dataset,only_spam])
    
dataset.shape


# In[15]:


#balanced dataset 
plt.figure(figsize=(8,8))
g = sns.countplot(x="label",data=dataset)
plt.title("Countplot for sam vs ham ")
plt.xlabel('Is the SMS spam?')
plt.ylabel('COunt')


# In[16]:


dataset['word_count'] = dataset['message'].apply(lambda x:len(x.split()))
dataset


# In[17]:


plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
g=sns.histplot(dataset[dataset['label']==0].word_count,kde=True)
p=plt.title("DISTRIBUTION OF WORD_COUNT FOR HAM SMS")

plt.subplot(1,2,2)
g=sns.histplot(dataset[dataset['label']==1].word_count,color="red",kde=True)
p=plt.title("DISTRIBUTION OF WORD_COUNT FOR SPAM SMS")

plt.tight_layout()
plt.show()


# In[18]:


#creating a new feature to check whether it has curreny symbols
def currency(dataset):
    currency_symbols = ['$','£','€','¥','₹']
    for i in currency_symbols:
        if i in dataset:
            return 1
    return 0


# In[19]:


dataset["Contains_currency_symbols"]=dataset["message"].apply(currency)
dataset


# In[20]:


plt.figure(figsize=(8,8))
g=sns.countplot(x="Contains_currency_symbols",data=dataset,hue="label")
plt.title("COUNTPLOT FOR CONTAING CURRENCY SYMBOL")
plt.xlabel("Does sms contain any currency symbol")
plt.ylabel("Count")
plt.legend(labels=["ham","spam"],loc=9)


# In[21]:


#creating new feature of containing numbers
def number (dataset):
    for i in dataset:
        if ord(i)>=48 and ord(i)<=57:
            return 1 
    return 0    
        


# In[22]:


dataset["contains_number"]=dataset["message"].apply(number)


# In[23]:


dataset


# In[24]:


plt.figure(figsize=(4,4))
g=sns.countplot(x="contains_number",data=dataset,hue="label")
plt.title("Countplot for containg numbers")
plt.xlabel("Does sms contain any numbers")
plt.ylabel("Count")
plt.legend(labels=["ham","spam"],loc=9)


# In[25]:


#Data cleaning
import nltk
import re
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[26]:


corpus = []
wnl = WordNetLemmatizer()

for sms in list(dataset.message):
    message = re.sub(pattern='[^a-zA-Z]',repl= ' ',string = sms)
    message = message.lower()
    words = message.split()#tokenization
    filtered_words = [word for word in message if word not in set(stopwords.words('english'))]
    lem_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ''.join(lem_words)
    
    
    corpus.append(message)


# In[27]:


corpus


# In[28]:


#building the bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features = 500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()


# In[29]:


x = pd.DataFrame(vectors,columns = feature_names)
y = dataset['label']


# In[30]:


from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# In[31]:


X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[32]:


X_train


# In[33]:


X_test


# In[34]:


Y_train


# In[35]:


Y_test


# In[36]:


#building naive bayes model
from sklearn.naive_bayes import MultinomialNB
mnb= MultinomialNB()
cv = cross_val_score(mnb,x,y,scoring='f1',cv=10)
print(round(cv.mean(),3))
print(round(cv.std(),3))


# In[37]:


mnb.fit(X_train,Y_train)
Y_pred = mnb.predict(X_test)


# In[38]:


print(classification_report(Y_test,Y_pred))


# In[39]:


cm = confusion_matrix(Y_test,Y_pred)
cm


# In[40]:


plt.figure(figsize=(8,8))
axis_labels= ['ham','spam']
g= sns.heatmap(data=cm,xticklabels=axis_labels,yticklabels=axis_labels,cmap="Blues")
p=plt.title("Cofusion matrix of Multinomial Naive Bayes Model")
plt.xlabel("Actual values")
plt.ylabel("Preducted values")


# In[41]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
cv1=cross_val_score(dt,x,y,scoring='f1',cv=10)
print(round(cv1.mean(),3))
print(round(cv1.std(),3))


# In[42]:


dt.fit(X_train,Y_train)
y_pred1=dt.predict(X_test)


# In[43]:


print(classification_report(Y_test,y_pred1))


# In[44]:


cm = confusion_matrix(Y_test,y_pred1)
cm


# In[45]:


plt.figure(figsize=(8,8))
axis_labels= ['ham','spam']
g= sns.heatmap(data=cm,xticklabels=axis_labels,yticklabels=axis_labels,cmap="Blues")
p=plt.title("Cofusion matrix of Multinomial Naive Bayes Model")
plt.xlabel("Actual values")
plt.ylabel("Preducted values")


# In[46]:


def predict_spam(sms):
    message = re.sub(pattern='[^a-zA-Z]',repl= ' ',string = sms)
    message = message.lower()
    words = message.split()#tokenization
    filtered_words = [word for word in message if word not in set(stopwords.words('english'))]
    lem_words = [wnl.lemmatize(word) for word in filtered_words]
    message = ''.join(lem_words)
    temp = tfidf.transform([message]).toarray()
    return mnb.predict(temp)


# In[47]:


#prediction 
sample_message='IMPORTANT - you have a chance to n lottery trip to paris for 4D3N'
if predict_spam(sample_message):
    print("THIS IS SPAM")
else :
    print("THIS IS HAM")


# In[48]:


#prediction2
sample2 = "I had never got any spam messages in my whole life"
if predict_spam(sample2):
    print("THIS IS SPAM")
else:
    print("THIS IS HAM")
    

