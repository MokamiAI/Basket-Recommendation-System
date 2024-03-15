#!/usr/bin/env python
# coding: utf-8

# # Shopping Basket Recommendation System with Python

# ## Introduction
# 
# Artificial intelligence is commonly used in various trade circles to automate processes, gather insights on business, and speed up processes. You will use Python to study the usage of artificial intelligence in real-life scenarios - how AI actually impacts businesses. 
# 
# Shopping basket basically contains a list of items bought by a person. A collection of such lists can be very informative for a shop since the data would indicate information like which products are in demand, what products are seasonal, etc. Businesses can identify which products need to be focused on and make recommendation based on the analysis. Shopping basket recommendation is the case where we can use AI to study the shopping list of a person and suggest to that person some things that he is likely to buy.
# 
# In this notebook, we will focus on shopping basket recommendation system using the KNN Model.
# 
# ## Context
# 
# We will be working with Amazon product reviews, obtained from [Kaggle](https://www.kaggle.com/saurav9786/recommender-system-using-amazon-reviews/data?select=ratings_Electronics+%281%29.csv). Kaggle is a platform for data enthusiasts to gather, share knowledge and compete for many prizes!
# 
# 
# 
# ## Customer Review Data
# 
# Big e-commerce companies like Amazon and Walmart deal with millions and millions of customers every day. The customers browse for products, buy them and sometimes leave reviews. Given that, the customer is the most important element for e-commerce companies; keeping them satisfied is primary.
# 
# Imagine that you know the shopping history of customers - what they buy and what are their preferences. You can use this information to your advantage by predicting what they might want to buy in the future and suggesting those things.
# 

# ### Side note: What is KNN?
# 
# KNN (K-Nearest Neighbors) is an algorithm used for both classification and regression. It assumes that similar things exist nearby, just like the saying, "Birds of a feather flock together." KNN algorithm classifies a new data point based on the class of its nearest neighbors, specifically k number of them. k denotes the number of nearest neighbors that helps in deciding the class of an object. The following diagram would make it clear:
# 
# 
# ![Knn where k = 3](https://cambridgecoding.files.wordpress.com/2016/01/knn2.jpg)

# ## Use Python to open csv files
# 
# We will use the [pandas](https://pandas.pydata.org/), [matplotlib](https://matplotlib.org/) and [scikit-learn](https://scikit-learn.org/stable/) libraries to work with our dataset. Pandas is a popular Python library for data science. It offers powerful and flexible data structures to make data manipulation and analysis easier. Scikit-learn is a very useful machine learning library that provides efficient tools for predictive data analysis. Matplotlib is a Python 2D plotting library that we can use to produce high quality data visualization. It is highly usable (as you will soon find out); you can create simple and complex graphs with just a few lines of codes!
# 
# ## Import Libraries
# 

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
get_ipython().run_line_magic('matplotlib', 'inline')


# Now that we have imported the libraries, let's start by reading the csv file.

# In[2]:


electronics_data=pd.read_csv(r"[Dataset]_Module11_(Recommendation).csv",names=['userId', 'productId','Rating','timestamp'])


# In[3]:


electronics_data.head(10)


# ### Task 1: Display the first 20 rows (11 to 20) of electronic data

# In[4]:


electronics_data.head(20)


# ## Getting information about dataset
# 
# If we can gather information about the dynamics of the datasets, that would give us a clear picture of the dataset and guide us in dealing with it.

# In[5]:


#Shape of the data
electronics_data.shape


# In[6]:


#Taking subset of the dataset
electronics_data=electronics_data.iloc[:1048576,0:]
print(electronics_data)


# In[7]:


#Check the datatypes
electronics_data.dtypes


# In[8]:


electronics_data.info()


# In[9]:


electronics_data.describe()['timestamp'].T


# ### Task 2: Display information about the Rating column in the dataset

# In[10]:


electronics_data.describe()['Rating'].T


# In[11]:


#Let us find the minimum and maximum ratings to find out whether ratings are in say 1-5 scale or 1-10 scale
print('Minimum rating is: %d' %(electronics_data.Rating.min()))
print('Maximum rating is: %d' %(electronics_data.Rating.max()))


# ### Task 3: Check for missing values in the dataset
# 

# In[12]:


print('numbers of missing values across columns: \n',electronics_data.isnull().sum() )


# In[13]:


# Let us check the distribution of the rating to find out different ratings distributions
with sns.axes_style('white'):
    g = sns.factorplot("Rating", data=electronics_data, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")


# In[14]:


print("Total data ")
print("_"*50)
print("\nTotal no of ratings :",electronics_data.shape[0])
print("Total No of Users   :", len(np.unique(electronics_data.userId)))
print("Total No of products  :", len(np.unique(electronics_data.productId)))


# ## Choosing only the dataset that we are interested in. 
# 
# Sometimes, we do not need the complete dataset for our estimations. Not all attributes of data may be useful for the model we are building. In that case, we can safely drop those attributes. For example, we won't need the timestamp column for our estimations here as it does not help us in any way in recommending the products that users might want to buy.

# In[15]:


#We are dropping the timestamp column here as we do not need it
electronics_data.drop(['timestamp'], axis=1,inplace=True)


# In[16]:


#Let us do an analysis of rating given by the user 

no_of_rated_products_per_user = electronics_data.groupby(by='userId')['Rating'].count().sort_values(ascending=False)

no_of_rated_products_per_user.head()


# ## Viewing the quantile distribution
# 
# Quantile is a point where a sample is divided into equally sized groups. A median of a sorted dataset is the middle point of that set where sorted means sorted in ascending or descending order. So, a median is a quantile as it divides the dataset into 2 equal groups.
# 
# ![Median(Quantile example)](https://www.statisticshowto.com/wp-content/uploads/2013/09/median.png)

# In[17]:


quantiles = no_of_rated_products_per_user.quantile(np.arange(0,1.01,0.01), interpolation='higher')


# In[18]:


plt.figure(figsize=(10,10))
plt.title("Quantiles and their Values")
quantiles.plot()
#We find quantiles with 0.05 difference
plt.scatter(x=quantiles.index[::5], y=quantiles.values[::5], c='orange', label="quantiles with 0.05 intervals")
#Let us also find quantiles with 0.25 difference
plt.scatter(x=quantiles.index[::25], y=quantiles.values[::25], c='m', label = "quantiles with 0.25 intervals")
plt.ylabel('No of ratings by user')
plt.xlabel('Value at the quantile')
plt.legend(loc='best')
plt.show()


# In[19]:


print('\n No of rated product more than 50 per user : {}\n'.format(sum(no_of_rated_products_per_user >= 50)) )


# ### Task 4: Display products where the number of users who rated is greater than 60

# In[20]:


print()


# ## Getting final working dataset based on popularity
# 
# We are importing the Surprise library which has the KNN function. We would see which products are really popular with the users and use those in recommending new items, as less popular items do not provide much intuition for recommendation.
# 
# If you do not have the library installed, please do the below step in your terminal: <br>
# pip install surprise

# In[21]:


import surprise
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD


# In[22]:


#POPULARITY BASED
#Let us get the new dataframe which contains users who have given 50 or more ratings. 
#You should try out the case where the new dataframe which contains users who have given 60 or more ratings
new_df=electronics_data.groupby("productId").filter(lambda x:x['Rating'].count() >=50)


# In[23]:


#Let us read the dataset here
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(new_df,reader)


# In[24]:


#We are splitting the dataset here
trainset, testset = train_test_split(data, test_size=0.3,random_state=10)


# In[25]:


algo = KNNWithMeans(k=5, sim_options={'name': 'pearson_baseline', 'user_based': False})
algo.fit(trainset)


# In[26]:


# Let us run the trained model against the testset
test_pred = algo.test(testset)


# ## Getting accuracy
# 
# Accuracy
# 
# The accuracy of a machine learning algorithm is a measure of how well the algorithm is performing - how  often the algorithm classifies a data point correctly. Accuracy is given by:
# 
# ![Accuracy](https://miro.medium.com/max/1050/1*O5eXoV-SePhZ30AbCikXHw.png)
# 
# Correlation Matrix
# 
# A correlation matrix is a table that shows the relation between variables- how one vaiable changes when another variable is changed. If there are 5 variables the correlation matrix will have 5 times 5 or 25 entries, where each entry shows the correlation between two variables.
# 
# RMSE
# 
# RMSE stands for root mean squared error. When we are doing predictions using our machine learning models, we need to find out if our predictions are correct. RMSE is a way of measuring the error in our predictions - if our RMSE is high, our predictions are bad and vice versa.

# In[ ]:


#We obtain the RMSE here
print("Item-based Model : Test Set")
accuracy.rmse(test_pred, verbose=True)


# Our final goal is to get a model that can predict. We could use existing user-item interactions to train a model to predict the top-5 items that might be the most suitable for a user. We will take the top 10000 recommendations and use an SVD to get the model.

# In[ ]:


new_df1=new_df.head(10000)
ratings_matrix = new_df1.pivot_table(values='Rating', index='userId', columns='productId', fill_value=0)
ratings_matrix.head()


# In[ ]:


X = ratings_matrix.T
X.head()


# In[ ]:


X1 = X
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)
decomposed_matrix.shape


# Finding the correlation matrix.

# In[ ]:


correlation_matrix = np.corrcoef(decomposed_matrix)
correlation_matrix.shape


# Suppose, we are considering the book with the id "B00000K135". We would find out the customer who is buying this book and recommend other books to him.

# In[ ]:


i = "B00000K135"

product_names = list(X.index)
product_ID = product_names.index(i)
product_ID


# Top items to be recommended to the customer who buys the item "B00000K135".

# In[ ]:


correlation_product_ID = correlation_matrix[product_ID]
correlation_product_ID.shape
Recommend = list(X.index[correlation_product_ID > 0.65])
# We are removing the item already bought by the customer
Recommend.remove(i) 
#Here we are printing the recommended items
Recommend[0:24]


# ### Task 5: Display recommendation for the customer who buys the item 'B00000JSGF'

# In[ ]:





# 
# ### Task 6: Display recommendation for the customer who buys the item 'B00000JDF6'

# In[ ]:


#yourcodehere


# ### Conclusion
# 
# Artificial intelligence is widely used by different modern-day industries to solve their problems. Here, in this notebook we have seen an example of how artificial intelligence can be used in the e-commerce industry by recommending items to customers based on their shopping habits.

# In[ ]:




