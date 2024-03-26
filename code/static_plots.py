#%%
## Importing necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns

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

# %%

## Handling the missing values in the dataset

## Dropping the missing row from release_date and title columns

movies_df.dropna(subset = ['title', 'release_date'], inplace = True)

## Imputing missing values in the following columns by mode
##genres, production_companies, production_countries and spoken_languages 

movies_df['genres', 'production_companies', 'production_countries', 'spoken_languages'].fillna(movies_df['genres', 'production_companies', 'production_countries', 'spoken_languages'].mode()[0], inplace = True)

print(f"First five rows of the dataset after dropping the missing values are:\n")
print(movies_df.head())

#%%

## Outlier Detection and Removal






# %%

