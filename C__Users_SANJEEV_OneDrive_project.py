#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


books = pd.read_csv('books.csv')
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')


# In[4]:


books.head()


# In[5]:


users.head()


# In[6]:


ratings.head()


# In[7]:


print(books.shape)
print(ratings.shape)
print(users.shape)


# In[8]:


books.isnull().sum()                   # to check for any null entries in books dataframe


# In[9]:


users.isnull().sum()


# In[10]:


ratings.isnull().sum()


# In[11]:


books.duplicated().sum()                # to check for any duplicate entries in the books dataframe


# In[12]:


ratings.duplicated().sum()


# In[13]:


users.duplicated().sum()


# # Popularity Based Book Recommender System
# 

# In[14]:


books_ratings = ratings.merge(books, on='ISBN')
books_ratings


# In[15]:


num_rating = books_ratings.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating.rename(columns={'Book-Rating':'num-ratings'},inplace=True)
num_rating


# In[16]:


avg_rating = books_ratings.groupby('Book-Title').mean()['Book-Rating'].reset_index()
avg_rating.rename(columns={'Book-Rating':'avg-ratings'},inplace=True)
avg_rating


# In[17]:


popular = num_rating.merge(avg_rating,on='Book-Title')
popular


# In[18]:


popular = popular[popular['num-ratings']>=250].sort_values('avg-ratings',ascending=False).head(50)
popular


# In[19]:


popular = popular.merge(books,on = 'Book-Title')
popular


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[21]:


#  plot of first 50 books 
sns.boxplot( x="avg-ratings", y='Book-Title', data=popular.head(50), )
plt.show()


# In[22]:


popular = popular.drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num-ratings','avg-ratings']]


# In[24]:


popular['Image-URL-M'][0]


# # Collaborative Filtering Based Book Recommender System
# 

# In[25]:


x = books_ratings.groupby('User-ID').count()['Book-Rating'] > 200
needed_users = x[x].index
needed_users


# In[26]:


filtered_rating = books_ratings[books_ratings['User-ID'].isin(needed_users)]
filtered_rating


# In[27]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index


# In[28]:


famous_books


# In[29]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[30]:


final_ratings


# In[31]:


# plot of book-title and book-ratings for first 1000 books
sns.boxplot( x="Book-Rating", y='Book-Title', data=final_ratings.head(1000), )
plt.show()


# In[32]:


pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')


# In[33]:


pt.fillna(0,inplace=True)


# In[34]:


pt


# In[35]:


pt.head(50)


# In[36]:


from sklearn.metrics.pairwise import cosine_similarity


# In[37]:


similarity_scores = cosine_similarity(pt)


# In[38]:


similarity_scores.shape


# In[39]:


similarity_scores 


# In[40]:


def recommend(book_name):
       # index fetch
       index = np.where(pt.index == book_name)[0][0]
       similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse = True)[1:6]
       
       for i in similar_items:
           print(pt.index[i[0]])


# In[41]:


recommend('A Time to Kill')


# In[ ]:





# In[ ]:





# In[ ]:




