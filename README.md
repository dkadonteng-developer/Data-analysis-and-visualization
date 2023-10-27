# Data-analysis-and-visualization
 Python code that performs various data analysis and visualization tasks on a dataset using the pandas, matplotlib, seaborn, and scikit-learn libraries.

Here's a breakdown of what the code does:

1. Imports the necessary libraries:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
```

2. Loads a dataset from a CSV file into a pandas DataFrame:
```
df = pd.read_csv('/Users/KNUST/Downloads/dataset.csv')
```

3. Displays the first few rows of the DataFrame:
```
df.head()
```

4. Displays information about the DataFrame, including column names, data types, and non-null counts:
```
df.info()
```

5. Prints the same information as above using the `print` function:
```
print(df.info())
```

6. Prints the shape of the DataFrame (number of rows and columns):
```
df.shape
```

7. Computes and displays descriptive statistics of the DataFrame:
```
df.describe()
```

8. Plots a histogram of the 'count' variable using matplotlib:
```
plt.figure(figsize=(8, 6))
plt.hist(df['count'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Total Rental Count')
plt.xlabel('Total Rental Count')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.75)
plt.show()
```

9. Checks for missing values in the DataFrame and prints the number of missing values for each column:
```
df.isnull().sum()
```

10. Replaces missing values in the 'count' column with the mean value of the column:
```
mean_count = df['count'].mean()
df['count'].fillna(mean_count, inplace=True)
```

11. Replaces missing values in the 'holiday' column with the mode (most frequent value) of the column:
```
mode_holiday = df['holiday'].mode().iloc[0]
df['holiday'].fillna(mode_holiday, inplace=True)
```

12. Computes the correlation matrix of the DataFrame and plots it as a heatmap using seaborn:
```
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()
```

13. Creates a scatter plot of the 'temp' variable against the 'count' variable using matplotlib:
```
plt.scatter(df['temp'], df['count'])
plt.title('Temperature vs Total Rental Count')
plt.xlabel('Temperature')
plt.ylabel('Total Rental Count')
plt.show()
```

14. Creates a box plot of the 'season' variable against the 'count' variable using seaborn:
```
sns.boxplot(x='season', y='count', data=df)
plt.title('Season vs Total Rental Count')
plt.xlabel('Season')
plt.ylabel('Total Rental Count')
plt.show()
```

15. Creates a bar plot of the 'weather' variable against the average 'count' using matplotlib:
```
df.groupby('weather')['count'].mean().plot(kind='bar')
plt.title('Weather vs Average Total Rental Count')
plt.xlabel('Weather')
plt.ylabel('Average Total Rental Count')
plt.show()
```

16. Creates a time series plot of the 'datetime' variable against the 'count' variable using matplotlib:
```
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['count'])
plt.title('Time Series of Total Rental Count')
plt.xlabel('Date and Time')
plt.ylabel('Total Rental Count')
plt.xticks(rotation=45)
plt.show()
```

17. Converts the 'datetime' column to a datetime object and extracts the hour component into a new 'hour' column:
```
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
```

18. Groups the data by hour and calculates the average 'count' for each hour, then plots it as a time series:
```
hourly_counts = df.groupby('hour')['count'].mean()
plt.figure(figsize=(12, 6))
plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-')
plt.title('Hourly Time Series of Total Rental Count')
plt.xlabel('Hour of the Day')
plt.ylabel('Average Total Rental Count')
plt.xticks(range(24))
plt.grid(True)
plt.show()
```

19. Extracts additional temporal features (day, month, year) from the 'datetime' column:
```
df['day'] = df['datetime'].dt.day
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year
```

The code you provided performs various data analysis and visualization tasks using different libraries in Python. Here's a breakdown of what the code does:

1. Imports the necessary libraries:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

2. Loads a dataset from a CSV file into a pandas DataFrame:
```
df = pd.read_csv('dataset.csv')
```

3. Displays the first few rows of the DataFrame:
```
print(df.head())
```

4. Displays the summary statistics of the DataFrame:
```
print(df.describe())
```

5. Checks for missing values in the DataFrame:
```
print(df.isnull().sum())
```

6. Replaces missing values in the 'Age' column with the mean age:
```
mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)
```

7. Replaces missing values in the 'Salary' column with the median salary:
```
median_salary = df['Salary'].median()
df['Salary'].fillna(median_salary, inplace=True)
```

8. Plots a histogram of the 'Age' column using matplotlib:
```
plt.hist(df['Age'], bins=10, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
```

9. Creates a bar plot of the 'Department' column:
```
department_counts = df['Department'].value_counts()
plt.bar(department_counts.index, department_counts.values)
plt.title('Department Distribution')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

10. Creates a scatter plot of the 'Age' against the 'Salary' column:
```
plt.scatter(df['Age'], df['Salary'])
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
```

11. Groups the data by the 'Department' column and calculates the mean salary for each department, then creates a bar plot:
```
department_salary_mean = df.groupby('Department')['Salary'].mean()
plt.bar(department_salary_mean.index, department_salary_mean.values)
plt.title('Mean Salary by Department')
plt.xlabel('Department')
plt.ylabel('Mean Salary')
plt.xticks(rotation=45)
plt.show()
```

12. Computes the correlation matrix of the DataFrame and plots it as a heatmap:
```
correlation_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

13. Splits the DataFrame into input features (X) and target variable (y):
```
X = df.drop('Salary', axis=1)
y = df['Salary']
```

14. Splits the data into training and testing sets:
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

15. Trains a linear regression model on the training data and makes predictions on the test data:
```
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

16. Computes and prints the mean squared error and R-squared score of the model:
```
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared Score:', r2)
```

17. Prints the coefficients of the linear regression model:
```
print('Coefficients:', model.coef_)
```

18. Plots the predicted values against the actual values for the test data:
```
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Actual vs Predicted')
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.show()
```

This code performs basic data exploration, missing value imputation, visualization, and linear regression modeling on a given dataset.
