#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
# Load the dataset
df = pd.read_csv('/Users/Derrick/Downloads/dataset.csv')


# In[8]:


df.head()


# In[9]:


df.info()


# In[10]:


print(df.info())


# In[11]:


df.shape


# In[14]:


df.describe()


# In[16]:


import matplotlib.pyplot as plt

# Plot a histogram of the 'count' variable
plt.figure(figsize=(8, 6))
plt.hist(df['count'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Rental Count')
plt.xlabel('Total Rental Count')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.75)
plt.show()


# In[17]:


df.isnull().sum()


# In[18]:


# Replace missing values in 'count' with the mean
mean_count = df['count'].mean()
df['count'].fillna(mean_count, inplace=True)


# In[19]:


# Replace missing values in 'holiday' with the mode
mode_holiday = df['holiday'].mode().iloc[0]
df['holiday'].fillna(mode_holiday, inplace=True)


# In[20]:


df.isnull().sum()


# In[25]:


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[21]:


import matplotlib.pyplot as plt

# Scatter plot of 'temp' vs 'count'
plt.scatter(df['temp'], df['count'])
plt.title('Temperature vs Total Rental Count')
plt.xlabel('Temperature')
plt.ylabel('Total Rental Count')
plt.show()


# In[22]:


import seaborn as sns

# Box plot of 'season' vs 'count'
sns.boxplot(x='season', y='count', data=df)
plt.title('Season vs Total Rental Count')
plt.xlabel('Season')
plt.ylabel('Total Rental Count')
plt.show()


# In[23]:


# Bar plot of 'weather' vs average 'count'
df.groupby('weather')['count'].mean().plot(kind='bar')
plt.title('Weather vs Average Total Rental Count')
plt.xlabel('Weather')
plt.ylabel('Average Total Rental Count')
plt.show()


# In[24]:


# Time series plot of 'datetime' vs 'count'
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['count'])
plt.title('Time Series of Total Rental Count')
plt.xlabel('Date and Time')
plt.ylabel('Total Rental Count')
plt.xticks(rotation=45)
plt.show()


# In[26]:


import matplotlib.pyplot as plt

# Convert the 'datetime' column to a datetime object if it's not already
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract the hour from the 'datetime' column
df['hour'] = df['datetime'].dt.hour

# Group the data by hour and calculate the average count for each hour
hourly_counts = df.groupby('hour')['count'].mean()

# Create a time series plot
plt.figure(figsize=(12, 6))
plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-')
plt.title('Hourly Time Series of Total Rental Count')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Total Rental Count')
plt.xticks(range(24))
plt.grid(True)
plt.show()


# In[27]:


# Convert the 'datetime' column to a datetime object if it's not already
df['datetime'] = pd.to_datetime(df['datetime'])

# Create new columns for 'hour,' 'day,' 'month,' and 'year'
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year

# Display the updated DataFrame
print(df.head())


# In[28]:


# Use one-hot encoding to encode categorical variables and add them to the DataFrame
df = pd.get_dummies(df, columns=['season', 'holiday', 'workingday', 'weather'], prefix=['season', 'holiday', 'workingday', 'weather'])

# Display the updated DataFrame with one-hot encoded columns
print(df.head())


# In[29]:


# Drop the specified columns from df_encoded
df_encoded = df_encoded.drop(['datetime', 'casual', 'registered'], axis=1)

# Display the updated DataFrame
print(df_encoded.head())


# In[30]:


# Drop the specified columns from df_encoded
df = df.drop(['datetime', 'casual', 'registered'], axis=1)

# Display the updated DataFrame
print(df.head())


# In[32]:


import numpy as np

# Apply a natural logarithm transformation to the 'count' column
df['count'] = np.log1p(df['count'])
df.head()


# In[33]:


print("Split Training and Test Data into 70-30")


# In[34]:


from sklearn.model_selection import train_test_split

# Separate the features (X) from the target variable (y)
X = df.drop('count', axis=1)
y = df['count']

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the shapes of the resulting datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[35]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[36]:


get_ipython().system('pip install xgboost')


# In[37]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV


# In[38]:


xgb_reg = xgb.XGBRegressor()


# In[39]:


param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5]
}


# In[40]:


grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[42]:


best_params = grid_search.best_params_
best_xgb_reg = grid_search.best_estimator
print("best estimator:",best_estimator)


# In[43]:


best_params = grid_search.best_params_
best_xgb_reg = grid_search.best_estimator
print("best estimator:",best_params)


# In[44]:


best_params = grid_search.best_params_
best_xgb_reg = grid_search.best_estimator
print("best_param:",best_params)


# In[45]:


from sklearn.metrics import mean_squared_error, r2_score

y_pred = best_xgb_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Hyperparameters:", best_params)
print("Mean Squared Error:", mse)
print("R-squared:", r2)


# In[ ]:




