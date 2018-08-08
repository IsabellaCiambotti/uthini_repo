
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier


# In[2]:


text = pd.read_csv('/Users/Isabella/Uthini/output.csv')


# In[3]:


text.sample(frac=1/2)


# In[4]:


text = text[text['channel_name'] != 'Project Dokotela Tutors'];


# <h3> fixing error where tutors were mislabeled </h3>

# In[5]:


#names = text['from'].unique()


# In[6]:


all_tutors = pd.read_csv('all_tutors.csv')


# In[7]:


all_tutors = all_tutors[all_tutors['Is Tutor'] == 'Yes'];


# In[8]:


text2 = text.copy()


# In[9]:


text2.head()


# In[10]:


text2.rename(columns={'from':'Full Name'}, inplace=True);


# In[11]:


all_tutors.head()


# In[12]:


names = pd.merge(text2, all_tutors, on='Full Name', how='inner')


# In[13]:


print(text2['Full Name'].nunique()) 
print(all_tutors['Full Name'].nunique())


# In[14]:


names['Full Name'].nunique()


# In[15]:


names = pd.Series(names['Full Name'].unique())


# In[16]:


names


# In[17]:


text['from'] = np.where(text['from'].isin(names),'TZ '+ text['from'] , text['from'])


# In[18]:


text.sample(frac=1/4)


# <h2> Type identification </h2>

# In[19]:


text['from_2'] = pd.np.where(text['from'].str.contains("TZ", case=True), "tutor",
                   pd.np.where(text['from'].str.contains("ThishaBot"), "bot",
                   pd.np.where(text['from'].str.contains("Setup"), "setup",
                   pd.np.where(text['from'].str.contains("Set up"), "setup",
                   pd.np.where(text['from'].str.contains("UthiniSupport"), "UthiniSupport", "student")))))


# In[20]:


text.drop(['type','type2', 'text2', 'edited'], axis=1, inplace=True)


# In[21]:


pd.set_option('display.max_colwidth', -1)
text.sample(frac=3/4)


# <h2> Naive Bayes Classification </h2>

# <i>using randomly sampled train data</i>

# In[22]:


#pretrain = text.sample(frac=1/16)


# In[23]:


#pretrain.to_csv('updated_train_small.csv', index=False)


# In[24]:


dtype={"type": object}
train_full = pd.read_csv('train_small.csv', dtype=dtype, encoding='utf-8')


# In[25]:


train_full.dropna(subset=['type'], inplace=True);
train_full.shape


# In[26]:


from textblob import TextBlob
import sklearn 
from textblob.classifiers import NaiveBayesClassifier


# In[27]:


train_full['text_type'] = tuple(zip(train_full['text'].astype(str), train_full['type'].astype(str)))


# In[30]:


test = train_full.iloc[210:260]
train = train_full.iloc[0:209]


# In[31]:


test.shape


# In[32]:


train_x = train['text_type'].tolist();
test_x = test['text_type'].tolist();


# In[33]:


cl = NaiveBayesClassifier(train_x)


# In[34]:


cl.accuracy(test_x)


# ## on full dataset

# In[35]:


dtype={"type": object}
train2 = pd.read_csv('train_small.csv', dtype=dtype)


# In[36]:


train2.dropna(subset=['type'], inplace=True);
train2.shape


# In[38]:


train2['text_type'] = tuple(zip(train2['text'].astype(str), train2['type'].astype(str)))


# In[39]:


test2 = text


# In[40]:


test2['type'] = np.nan


# In[41]:


test2['text_type'] = tuple(zip(test2['text'].astype(str), test2['type'].astype(str)))


# In[42]:


train_2x = train2['text_type'].tolist();
test_2x = test2['text_type'].tolist();


# In[43]:


cl = NaiveBayesClassifier(train_2x)


# In[ ]:


cl.accuracy(test_2x)


# In[ ]:


#test = pd.merge(text, train, on='id', how='left', indicator=True)


# In[ ]:


#test = test[test['type'].isnull()]


# In[ ]:


#test['from_2_y'].isnull().sum()


# In[ ]:


#test.drop(['channel_type_y','channel_name_y', 'date_y', 'from_y', 'text_y', 'from_2_y','_merge'], axis=1, inplace=True)


# In[ ]:


#test.rename(columns={'channel_type_x':'channel_type', 'channel_name_x':'channel_name', 'date_x': 'date', 'from_x':'from', 'text_x':'text', 'from_2_x':'from_2'}, inplace=True)

