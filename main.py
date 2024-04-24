#%%
## Importing necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

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

## Outlier Detection

fig, axes = plt.subplots(3, 2, figsize=(12, 8))

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


#%%
## Normality Detection

## Checking the normality of vote_average, vote_count, revenue, runtime, budget and popularity columns using histogram plot.

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Specify the columns for histograms
columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']

# Iterate over each column and plot the histogram
for i, column in enumerate(columns):
    sns.histplot(x=movies_df[column], kde=True, ax=axes[i], bins=30)
    axes[i].set_title(f"Distribution of {column}")

# Adjust layout
plt.tight_layout()
plt.show()


# %%

## Checking normality of runtime column using KS test

ks_stat_runtime, p_val_runtime = st.kstest(movies_df['runtime'], 'norm')


def interpret_ks_test(p_val):
    if p_val >= 0.05:
        return 'Normal'
    else:
        return 'Not Normal'


print(f"K-S test : statistics = {round(ks_stat_runtime, 2)}, p-value = {round(p_val_runtime, 2)}")
print(f"K-S test : Runtime Column looks {interpret_ks_test(p_val_runtime)}")



# %%

# Data Transformation - Transforming Non - Normal Distributions to Normal Distributions.

# Applying Box Cox transformations to revenue, runtime, budget and popularity variables.

# Apply Box-Cox transformation to 'revenue' column
movies_df['revenue_transformed'], revenue_lambda = st.boxcox(movies_df['revenue'])

# Apply Box-Cox transformation to 'runtime' column
movies_df['runtime_transformed'], runtime_lambda = st.boxcox(movies_df['runtime'])

# Apply Box-Cox transformation to 'budget' column
movies_df['budget_transformed'], budget_lambda = st.boxcox(movies_df['budget'])

# Apply Box-Cox transformation to 'popularity' column
movies_df['popularity_transformed'], popularity_lambda = st.boxcox(movies_df['popularity'])

# You can print out the lambda values if needed
print("Lambda values:")
print("Revenue:", revenue_lambda)
print("Runtime:", runtime_lambda)
print("Budget:", budget_lambda)
print("Popularity:", popularity_lambda)


## Applying log(x + 1) transformation for vote_count column

movies_df['vote_count_transformed'] = np.log1p(movies_df['vote_count'])

# %%

## Plotting the histograms to show distribution of data after performing transformations on the data

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

# Flatten the axes array for easier iteration
axes = axes.flatten()

# Specify the columns for histograms
columns = ['vote_average', 'vote_count_transformed', 'revenue_transformed', 'runtime_transformed', 'budget_transformed', 'popularity_transformed']

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
plt.figure(figsize=(8, 5))
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

# %%

## Pearson Correlation Coefficient

## Selecting the numerical columns
numerical_columns = ['vote_average', 'vote_count', 'runtime', 'budget', 'popularity', 'revenue']

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

# %%
#
# numeric_columns = ['vote_average', 'vote_count', 'revenue', 'runtime', 'budget', 'popularity']
# numeric_df = movies_df[numeric_columns]
#
# # Plot the KDE pairplot with title
# sns.pairplot(numeric_df, diag_kind='kde')
# plt.suptitle('Multivariate KDE Pairplot', y=1.02)
# plt.show()

#%%

## Static Plots

## 1.) Bar Chart showing the top 10 movies with the most revenue.

# Sorting the dataset in descending order of revenue
sorted_data = movies_df.sort_values(by='revenue', ascending=False)
# Get the top 10 movies with highest revenue
top_10_movies = sorted_data.head(10)

# Plotting the bar plot
plt.figure(figsize=(15, 12))
bars = plt.bar(top_10_movies['title'], top_10_movies['revenue'], color='skyblue')

# Adding revenue labels on top of each bar
for bar, revenue in zip(bars, top_10_movies['revenue']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'${revenue:,.0f}',
             ha='center', va='bottom')

plt.xlabel('Movie Title', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
plt.ylabel('Revenue', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
plt.title('Top 10 Movies with the Highest Revenue', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 20})
plt.xticks(rotation=45, ha='right', fontname='serif', size=12)
plt.tight_layout()
plt.show()

# %%

## 2.) Horizontal Bar Chart Showing the Top 5 Popular Movies with Average Votes and Vote_Count

# Sort the DataFrame by 'popularity' in descending order
sorted_data = movies_df.sort_values(by='popularity', ascending=False)

# Get the top 5 most popular movies
top_5_popular = sorted_data.head(5)

# Reverse the order for 'barh' to plot with the most popular at the top
top_5_popular = top_5_popular.iloc[::-1]

# Plotting
plt.figure(figsize=(15, 12))
bars = plt.barh(top_5_popular['title'], top_5_popular['popularity'], color='skyblue')
plt.xlabel('Popularity', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
plt.ylabel('Movie Title', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
plt.title('Top 5 Popular Movies with Popularity Score',
          fontdict={'fontname': 'serif', 'color': 'blue', 'size': 20})

for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():.2f}',
             va='center', ha='left', color='black', fontsize=10)

plt.tight_layout()
plt.show()

# %%

# ## 3.) Line Chart showing the total revenue for each year.

# Convert release_date to datetime
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'])

# Extract year from release_date
movies_df['year'] = movies_df['release_date'].dt.year

# Group by the year to calculate total revenue and total budget per year
yearly_totals = movies_df.groupby('year').agg({'revenue':'sum', 'budget':'sum'}).reset_index()

# Plot the area chart
plt.figure(figsize=(12, 6))
plt.fill_between(yearly_totals['year'], yearly_totals['revenue'], alpha=0.5, label='Revenue')  # Adjust alpha for transparency
plt.fill_between(yearly_totals['year'], yearly_totals['budget'], alpha=0.5, label='Budget')  # Adjust alpha for transparency
plt.title('Total Revenue and Budget by Year',
          fontdict={'fontname': 'serif', 'color': 'blue', 'size': 20})
plt.xlabel('Year',fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
plt.ylabel('Total Amount',fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%

## 4.) Subplot showing a count plot of movies by genre and a grouped bar chart showing the total budget and revenue for different genres.

# Splitting the genres and expanding them into separate rows
genres_expanded = movies_df.drop('genres', axis=1).join(
    movies_df['genres'].str.split(', ').explode().reset_index(drop=True),
    how='right'
)

# Counting movies by genre
genre_counts = genres_expanded['genres'].value_counts().sort_index()

# Calculating total budget and revenue by genre
genre_financials = genres_expanded.groupby('genres')[['budget', 'revenue']].sum().sort_index()

# Creating subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))

# Plotting the count of movies by genre
sns.barplot(x=genre_counts.index, y=genre_counts.values, ax=axes[0])
axes[0].set_title('Count of Movies by Genre', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 20})
axes[0].set_ylabel('Number of Movies', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
axes[0].set_xlabel('Genre', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
axes[0].set_xticklabels(genre_counts.index, rotation=45, ha='right', fontname='serif',
                        size=12)  # Adjusting x-tick labels

# Plotting total budget and revenue by genre
genre_financials.plot(kind='bar', ax=axes[1])
axes[1].set_title('Total Budget and Revenue by Genre', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 20})
axes[1].set_ylabel('Total Amount', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})
axes[1].set_xlabel('Genre', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 16})

# Correcting the x-ticks position for the grouped bar chart
axes[1].set_xticks(range(len(genre_financials.index)))
axes[1].set_xticklabels(genre_financials.index, rotation=45, ha='right', fontname='serif',
                        size=12)  # Adjusting x-tick labels

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()

# %%

## 5.) Regression Subplot showing the influence of Budget and Revenue on Popularity of movies

# Filter the DataFrame
df_plot = movies_df[(movies_df['budget'] != 0) & (movies_df['revenue'] != 0)]

# Creating the subplot
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

plt.suptitle('The Influence of Budget and Revenue on Popularity of Movies', fontsize=18, weight='bold', color='#333d29')

# Looping over the columns budget and revenue to create two subplots
for i, col in enumerate(['budget', 'revenue']):
    sns.regplot(data=df_plot, x=col, y='popularity',
                scatter_kws={"color": "#06837f", "alpha": 0.6}, line_kws={"color": "#fdc100"}, ax=axes[i])
    # Setting title for each subplot
    axes[i].set_title(col.capitalize() + ' vs Popularity', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 14})
    # Setting x and y labels with specified font properties
    axes[i].set_xlabel(col.capitalize(), fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 12})
    axes[i].set_ylabel('Popularity', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 12})

# Adjust layout to make space for the super title
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Display the plots
plt.show()

# %%

## 6.) Subplots showing the top 5 Production Countries based on the total number of movies and the total revenue generated


movies_df['production_countries'] = movies_df['production_countries'].apply(
    lambda x: x if isinstance(x, list) else x.split(', '))

# Explode the 'production_countries' DataFrame
exploded_countries_df = movies_df.explode('production_countries')

production_countries_count = exploded_countries_df['production_countries'].value_counts().nlargest(5)

production_countries_revenue = exploded_countries_df.groupby('production_countries')['revenue'].sum().nlargest(5)

# Creating a subplot with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(20, 10))

# Subplot for production countries by number of movies
country_bars_count = axs[0].bar(range(len(production_countries_count)), production_countries_count.values)
axs[0].set_title('Top 5 Production Countries by Total Number of Movies',
                 fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
axs[0].set_xlabel('Production Countries', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
axs[0].set_ylabel('Number of Movies', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
axs[0].set_xticks(range(len(production_countries_count)))
axs[0].set_xticklabels(production_countries_count.index, rotation=45, ha='right', fontsize=8)

country_bars_revenue = axs[1].bar(range(len(production_countries_revenue)), production_countries_revenue.values)
axs[1].set_title('Top 5 Production Countries by Revenue', fontdict={'fontname': 'serif', 'color': 'blue', 'size': 16})
axs[1].set_xlabel('Production Countries', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
axs[1].set_ylabel('Revenue', fontdict={'fontname': 'serif', 'color': 'darkred', 'size': 14})
axs[1].set_xticks(range(len(production_countries_revenue)))
axs[1].set_xticklabels(production_countries_revenue.index, rotation=45, ha='right', fontsize=8)

plt.tight_layout()
plt.show()

# %%

## Adult Versus Non Adult Analysis : Adult vs. Non-Adult Analysis: A table comparing average revenue, budget, and popularity between adult and non-adult films.\

# Grouping by 'adult' and calculating average revenue, average budget, and average popularity
avg_stats = movies_df.groupby('adult').agg({'revenue': 'mean', 'budget': 'mean', 'popularity': 'mean'}).reset_index()

# Creating a PrettyTable instance
table = PrettyTable()

# Defining column names
table.field_names = ['Adult', 'Average Revenue', 'Average Budget', 'Average Popularity']

# Adding rows to the table
for index, row in avg_stats.iterrows():
    table.add_row(['Adult' if row['adult'] else 'Non-Adult', f"${row['revenue']:.2f}", f"${row['budget']:.2f}",
                   f"{row['popularity']:.2f}"])

# Printing the table
print(table)

#%%



