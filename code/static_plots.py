#%%
## Importing necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels. graphics.gofplots import qqplot
import scipy.stats as st

# %%
url ='https://drive.google.com/file/d/1izawoE4HceqSWJBMwZ-cv7f0-50tBD-l/view?usp=sharing'
movies_df=pd.read_csv('https://drive.usercontent.google.com/download?id={}&export=download&authuser=0&confirm=t'.format(url.split('/')[-2]))


#%%
## Checking rows and columns of dataset

print(movies_df.shape)

#%%
## Printing the first five rows of the dataset
print(f"First five rows of the dataset are : {movies_df.head()}")

# %%
##Information about dataset

print("Information about the dataset:\n")
print(movies_df.info())

#%%

## Five point summary statistics of numerical columns

print("Five point summary of the dataset is: \n")
print(movies_df.describe())

# %%

## Dropping the unecessary columns from the dataset

columns = ['backdrop_path', 'homepage', 'imdb_id', 'tagline','poster_path','overview','original_title','original_language']

movies_df.drop(labels= columns, axis=1, inplace = True)

print(f"First five rows after dropping the rows are: {movies_df.head()}")

#%%
## Checking if there are any missing values in dataset

if movies_df.isnull().values.any or movies_df.isna().values.any:
    print("There are missing values in the dataset")
else:
    print("There are no missing values in the dataset")

missing_values = movies_df.isnull().sum()

print(f"Missing values in the dataset are: {missing_values}")

# Dropping the missing row from release_date and title columns
movies_df.dropna(subset=['title', 'release_date'], inplace=True)

# Imputing missing values in the following columns by mode
for column in ['genres', 'production_companies', 'production_countries', 'spoken_languages']:
    mode_value = movies_df[column].mode()[0]
    movies_df[column].fillna(mode_value, inplace=True)

# Displaying the first five rows of the dataset after handling the missing values
print("First five rows of the dataset after handling the missing values are:\n")
print(movies_df.head())

#%%
missing_values =  movies_df.isnull().sum()
print("Count of missing values after cleaning the dataset\n")
print(missing_values)

#%%

## Detecting outliers for vote_average column using boxplot

plt.figure(figsize = (10,8))
plt.boxplot(movies_df['vote_average'])
plt.title('Distribution of Vote_Average')
plt.xlabel('Vote_Average')
plt.tight_layout()
plt.show()

## From the above boxplot, we can see that there are no outliers in the vote_average column.

#%%

## Revenue cannot be less than zero, so checking if there are any rows in the dataset having revenue value less than 0.

rev_lt_zero = movies_df[movies_df['revenue'] < 0]
print(rev_lt_zero)

## We can see that there is one record having revenue as - 12. We will drop this row from the dataset.
movies_df = movies_df.drop(rev_lt_zero.index)

## Further examining the vote_count, runtime, budget and popularity columns, these columns have higher values but they appear to be valid data points.
## Therefore, I decided to retain the original data rather than remove those outliers.

# %%

## Normality Detection

## Using QQ Plot

## Using QQ Plot to check normality in vote_average column

plt.figure(figsize = (10, 8))
qqplot(movies_df['vote_average'], line = 's')
plt.title('Q-Q Plot of Vote_Average')
plt.show()

## At the left tail, the points fall below the line, suggesting that the data has fewer extreme low values than a normal distribution would predict.
## We can also see that at the right tail, there are more extreme high values than would be expected if the data were normally distributed.
## The steps in the plot suggest that there are a lot of data points with the same or similar values.
## Overall, the plot tell us that the vote_average column is not normally distributed.

#%%
## Using QQ plot to check normality in vote_count column 

plt.figure(figsize = (10, 8))
qqplot(movies_df['vote_count'], line = 's')
plt.title('Q-Q Plot of Vote_Count')
plt.show()

## From the QQ plot of vote_count column, we can see that the data is higly skewed. 
## The extreme deviation from the line in the upper tail indicates the presence of outliers.
## We can see that the vote_count column does not follow a normal distribution.

# %%

plt.figure(figsize = (10, 8))
qqplot(movies_df['revenue'], line = 's')
plt.title('Q-Q Plot of Revenue')
plt.show()

## The QQ plot shows the normality of Revenue column. 
## The plot shows a strong right skewed distribution with a long tail, suggesting the presence of very high revenue values compared to what would be expected in a normal distribution.
## This indicates that the data in the revenue column is not normally distributed.

# %%

## Checking normality of runtime column using KS test

ks_stat_runtime, p_val_runtime = st.kstest(movies_df['runtime'], 'norm')

def interpret_ks_test(p_val):

    if p_val >= 0.05:
        return 'Normal'
    else:
        return 'Not Normal'

print(f"K-S test : statistics = {round(ks_stat_runtime,2)}, p-value = {round(p_val_runtime, 2)}")
print(f"K-S test : Runtime Column looks {interpret_ks_test(p_val_runtime)}")

## The test statistic 0.77 is quite large, suggesting a substantial difference between the runtime column distribution and normal distribution.
## The p value of 0 indicates that the difference is statistically significant.
## The test strongly rejects the null hypothesis that the data comes from a normal distribution.

# %%

## Checking normality of budget column using KS test

ks_stat_budget, p_val_budget = st.kstest(movies_df['budget'], 'norm')

def interpret_ks_test(p_val):

    if p_val >= 0.05:
        return 'Normal'
    else:
        return 'Not Normal'

print(f"K-S test : statistics = {round(ks_stat_budget,2)}, p-value = {round(p_val_budget, 2)}")
print(f"K-S test : Budget Column looks {interpret_ks_test(p_val_budget)}")

## The test statistic is 0.5 is large, suggesting a difference between the budget column distribution and normal distribution.
## The p value of 0 indicates that the difference is statistically significant.
## The test strongly rejects the null hypothesis that the data comes from a normal distribution.

# %%

## Checking normality of popularity column using KS test

ks_stat_popularity, p_val_popularity = st.kstest(movies_df['popularity'], 'norm')

def interpret_ks_test(p_val):

    if p_val >= 0.05:
        return 'Normal'
    else:
        return 'Not Normal'

print(f"K-S test : statistics = {round(ks_stat_popularity,2)}, p-value = {round(p_val_popularity, 2)}")
print(f"K-S test : Popularity Column looks {interpret_ks_test(p_val_popularity)}")

## The test statistic is 0.69 is large, suggesting a difference between the budget column distribution and normal distribution.
## The p value of 0 indicates that the difference is statistically significant.
## The test strongly rejects the null hypothesis that the data comes from a normal distribution.

# %%

