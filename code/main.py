#%%
## Importing necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels. graphics.gofplots import qqplot
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

## Dropping rows from the dataset having values less than zero or zero

movies_df = movies_df[(movies_df['budget'] > 0)]
movies_df['revenue'] = movies_df['revenue'].replace(0, movies_df['revenue'].mean())
movies_df['runtime'] = movies_df['runtime'].replace(0, movies_df['runtime'].median())
movies_df['popularity'] = movies_df['popularity'].replace(0, movies_df['popularity'].median())

#%%

## Revenue cannot be less than zero, so checking if there are any rows in the dataset having revenue value less than 0.

rev_lt_zero = movies_df[movies_df['revenue'] < 0]
print(rev_lt_zero)

## We can see that there is one record having revenue as - 12. We will drop this row from the dataset.
movies_df = movies_df.drop(rev_lt_zero.index)

#%%

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Specify the columns for boxplots
columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget','popularity']

# Iterate over each column and plot the boxplot
for i, column in enumerate(columns):
        sns.boxplot(x=movies_df[column], ax=axes[i])
        axes[i].set_title(f"Distribution of {column}")

# Adjust layout
plt.tight_layout()
plt.show()

## From the boxplots, we can see that the vote_average column does not have any outliers.
##  The boxplots of vote_count, revenue, runtime, budget and popularity display outliers on the higher ends.
## However, these outliers appear to be valid data points.
## Therefore, I decided to retain the original data rather than remove those outliers.

# %%

## Normality Detection

## Checking the normality of vote_average, vote_count, revenue, runtime, budget and popularity columns using histogram plot.

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Specify the columns for histograms
columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']

# Iterate over each column and plot the histogram
for i, column in enumerate(columns):
    sns.histplot(x=movies_df[column], kde=True, ax=axes[i], bins = 30)
    axes[i].set_title(f"Distribution of {column}")


# Adjust layout
plt.tight_layout()
plt.show()

## The distribution of vote_average appears to be left skewed with a concentration of values, with a concentration of values towards the higher end of the scale and a long tail extending towards the lower scores.
## This indicates that most of the movies receive a high average vote, with fewer movies receiving low ratings.

## The distribution of vote_count shows a highly right skewed distribution. 
## There are a significant number of movies with very few votes and the frequency quickly drops off as the number of votes increases, suggesting that few movies receive a large number of votes.

## The revenue distribution is also right skewed. Most movies earn a relatively small amount of revenue, while the counts decreases as the revenue increases. High earning movies are relatively rare.

## The runtime distribution is also right skewed with a few movies with very long runtime that creates longer tails.

## Similar to revenue, the budget is also right skewed. Most movies have lower budgets, with fewer movies having very high budgets.

## The popularity distribution is highly right skewed, with most movies having low popularity scores and a very quick drop off as popularity increases. This implies that only a small number of movies achieve higher popularity.

## Overall, we can see that none of the variables are normally dsitributed. We need to transform these variables from Non-Gaussian to Gaussian distribution.

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

## The test statistic 0.99 is quite large, suggesting a substantial difference between the runtime column distribution and normal distribution.
## The p value of 0 indicates that the difference is statistically significant.
## The test strongly rejects the null hypothesis that the data comes from a normal distribution.

# %%

## Data Transformation - Transforming Non - Normal Distributions to Normal Distributions.

## Applying Box Cox transformations to revenue, runtime, budget and popularity variables.

# Apply Box-Cox transformation to 'revenue' column
movies_df['revenue'], revenue_lambda = st.boxcox(movies_df['revenue']) 

# Apply Box-Cox transformation to 'runtime' column
movies_df['runtime'], runtime_lambda = st.boxcox(movies_df['runtime'])  

# Apply Box-Cox transformation to 'budget' column
movies_df['budget'], budget_lambda = st.boxcox(movies_df['budget'])  

# Apply Box-Cox transformation to 'popularity' column
movies_df['popularity'], popularity_lambda = st.boxcox(movies_df['popularity'])  

# You can print out the lambda values if needed
print("Lambda values:")
print("Revenue:", revenue_lambda)
print("Runtime:", runtime_lambda)
print("Budget:", budget_lambda)
print("Popularity:", popularity_lambda)


## Applying log(x + 1) transformation for vote_count column

movies_df['vote_count'] = np.log1p(movies_df['vote_count'])

# %%

## Plotting the histograms to show distribution of data after performing transformations on the data

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Specify the columns for histograms
columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']

# Iterate over each column and plot the histogram
for i, column in enumerate(columns):
    sns.histplot(x=movies_df[column], kde=True, ax=axes[i], bins = 30)
    axes[i].set_title(f"Distribution of Transformed {column}")


# Adjust layout
plt.tight_layout()
plt.show()

# %%

## Applying PCA to the variables

## Selecting the numerical columns as features
features = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']

# Standardizing the features
x = movies_df.loc[:, features].values
x = StandardScaler().fit_transform(x)

# Applying PCA
pca = PCA()
principalComponents = pca.fit(x)

## Checking the condition number and getting the singular values 
condition_number = np.linalg.cond(pca.components_)
singular_values = pca.singular_values_

print(f"Condition Number: {condition_number}")
print(f"Singular Values: {singular_values}")

# Plot cumulative explained variance
plt.figure(figsize=(8,5))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance by PCA Components')
plt.grid(True)
plt.show()

# Determine how many features can be removed
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components_for_90 = np.where(cumulative_variance >= 0.9)[0][0] + 1
features_to_remove = len(features) - num_components_for_90

print(f"Number of components to retain 90% of variance: {num_components_for_90}")
print(f"Features that can be removed: {features_to_remove}")

## Explanation of PCA:

## Condition Number : The condition number of the PCA components matrix is approximately 1, which indicates that the matrix is well conditioned. 
## This means that there is no significant multicollinearity in the data and suggests that the principal components are numerically stable.

## Singular Values : The first singular value is 383.54, which is significantly higher than the rest. This indicates that the first principal component captures a substantial amount of variance within the dataset.
## As the singular value decreases, they indicate principal components with diminishing contributions to capturing the dataset's variance.
## The second singular value is 200.02, which is roughly half of the first, meaning that while substantial, it contributes significantly less to explaining the variance than the first component. This pattern continues with the third, fourth, fifth and sixth singular values.

## Cumulative Explained Variance Plot : From the cumulative explained variance plot, we see that 4 components are needed to retain 90% of the variance.
## This means that the first four principal components capture most of the information that was contained in the original six features.
## Since, we can capture 90% of the variance with 4 components, we can reduce the feature space from 6 to 4, effectively removing 2 features. 

#%%

## Pearson Correlation Coefficient

## Selecting the numerical columns 
numerical_columns = ['vote_average', 'vote_count', 'runtime', 'budget', 'popularity','revenue']

# Calculating correlation matrix
correlation_matrix = movies_df[numerical_columns].corr()

## Visualizing correlation matrix using heatmap

# Visualizing correlation matrix using heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Pearson Correlation Coefficient Heatmap')
plt.show()

# Plotting scatter plot matrix
plt.figure(figsize=(15, 12))
sns.pairplot(movies_df[numerical_columns])
plt.suptitle('Scatter Plot Matrix')
plt.show()

## Heatmap interpretation:

## The heatmap shows that revenue has the highest positive correlation with budget and popularity. This suggests that films with higher budgets and those that are more popular are likely to have higher revenue, but the relationship is stronger with budget.
## Revenue has very low correlation with vote_average and runtime. This implies that there's almost no linear relationship between the average rating or length of a movie and its runtime.
## Vote_Count has a moderate positive correlation with revenue, indicating that the movies with more votes tend to have higher revenue, though this relationship is not as strong as budget.

## Scatter Plot Matrix Interpretation:

## The scatter plot of revenue versus vote_count shows some positive trend; however, it does not seem to be a very strong linear relationship. There are movies with a high number of votes that vary widely in nature.
## There is a positive trend between revenue and budget, with higher budget movies tending to have higher revenue.
## There is a positive trend, but many data points are clustered at the lower end of popularity, suggesting that popularity alone isn't a strong predictor of revenue.
## There is not a clear pattern or trend between vote_average and revenue. The points are widely spread out, suggesting that for any given average vote, the corresponding revenues can vary greatly. This dispersion tells us that vote_average is not a strong determinant of a movie's revenue.
## The relationship between revenue and runtime appears weak,the plot shows a wide dispersion of revenue across various runtimes, suggesting that the length of a movie does not strongly predict its financial success.


#%%

## Static Plots










# %%
