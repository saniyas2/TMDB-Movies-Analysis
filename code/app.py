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
## Dropping the unecessary columns from the dataset

columns = ['backdrop_path', 'homepage', 'imdb_id', 'tagline','poster_path','overview','original_title','original_language']

movies_df.drop(labels= columns, axis=1, inplace = True)

print(f"First five rows after dropping the rows are: {movies_df.head()}")

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

# Convert release_date to datetime
movies_df['release_date'] = pd.to_datetime(movies_df['release_date'])

# Extract year from release_date
movies_df['year'] = movies_df['release_date'].dt.year

# Split 'production_companies' into multiple rows
movies_df['production_companies'] = movies_df['production_companies'].str.split(', ')
movies_df_exploded = movies_df.explode('production_companies')

# Calculate the number of movies per production company
movie_counts = movies_df_exploded['production_companies'].value_counts().reset_index()
movie_counts.columns = ['production_companies', 'movie_count']

# Calculate the total revenue by production company
revenue_by_company = movies_df_exploded.groupby('production_companies')['revenue'].sum().reset_index()
revenue_by_company.columns = ['production_companies', 'total_revenue']

# Aggregate revenue by month
# Convert 'release_date' to datetime and extract month as a name
movies_df['release_month_name'] = movies_df['release_date'].dt.strftime('%b')
movies_df['release_month'] = movies_df['release_date'].dt.month


# Extract month names from 'release_date'
movies_df['release_month_name'] = movies_df['release_date'].dt.strftime('%b')

# Aggregate revenue by month names
monthly_revenue = movies_df.groupby('release_month_name')['revenue'].sum().reset_index()

# Sort monthly_revenue by the actual order of the months
months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_revenue['release_month_name'] = pd.Categorical(monthly_revenue['release_month_name'], categories=months_order, ordered=True)
monthly_revenue = monthly_revenue.sort_values('release_month_name')

#%%

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("TMDB Movies Analysis", style={'textAlign': 'center', 'color': 'dark blue'}),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label = 'Total Number of Movies by Production Company', value='tab-1'),
        dcc.Tab(label = 'Total Revenue by Production Company', value='tab-2'),
        dcc.Tab(label = 'Total Revenue by Month', value='tab-3'),
        dcc.Tab(label = 'Total Revenue by Year', value='tab-4'),
        dcc.Tab(label = 'Total Number of Movies by Status', value='tab-5'),
    ]),
    html.Div(id='layout')
])

## Defining layout for Tab 1
tab1_layout = html.Div([
    dcc.Dropdown(
        id='production-company-dropdown',
        options=[{'label': i, 'value': i} for i in movie_counts['production_companies'].unique()],
        value=None,
        multi=True
    ),
    dcc.Graph(id='movie-count-bar-chart')
])

## Defining layout for Tab 2
tab2_layout = html.Div([
    dcc.Dropdown(
        id='production-company-dropdown-tab2',
        options=[{'label': i, 'value': i} for i in revenue_by_company['production_companies'].unique()],
        value=None,
        multi=True
    ),
    dcc.Graph(id='revenue-bar-chart')
])

## Defining layout for Tab 3

tab3_layout = html.Div([
    html.H3('Select Months:'),
    dcc.Checklist(
        id='month-checklist',
        options=[{'label': str(month), 'value': str(month)} for month in monthly_revenue['release_month_name']],
        value=['Jan'],  # Default to January
        labelStyle={'display': 'inline-block'}
    ),
    dcc.Graph(id='revenue-chart')
])


## Defining layout for Tab 4

tab4_layout = html.Div([
    html.H3('Select Year Range:'),
    dcc.RangeSlider(
        id='year-range-slider',
        min=movies_df['year'].min(),
        max=movies_df['year'].max(),
        step=5,
        marks={i: str(i) for i in range(int(movies_df['year'].min()), int(movies_df['year'].max())+1, 5)},
        value=[movies_df['year'].min(), movies_df['year'].max()]
    ),
    dcc.Graph(id='revenue-by-year-chart')
])


## Defining layout for Tab 5
tab5_layout = html.Div([
    html.Div([
        html.H3('Select Movie Statuses:'),
        dcc.Checklist(
            id='status-checklist',
            options=[{'label': status, 'value': status} for status in movies_df['status'].unique()],
            value=['Released'],  # Default to 'Released'
            labelStyle={'display': 'inline-block'}  # Added textAlign here
        ),
        dcc.Graph(id='status-pie-chart')
    ], style={'textAlign': 'center'})  # Added style here to center all content
])

# Define callback for updating tabs
@app.callback(Output('layout', 'children'), Input('tabs', 'value'))
def update_tabs(tab):
    if tab == 'tab-1':
        return tab1_layout
    elif tab == 'tab-2':
        return tab2_layout
    elif tab == 'tab-3':
        return tab3_layout
    elif tab == 'tab-4':
        return tab4_layout
    elif tab == 'tab-5':
        return tab5_layout
    else:
        return html.H1('Tab not implemented', style={'color': 'red'})

# Define callback for updating Tab 1 content
@app.callback(
    Output('movie-count-bar-chart', 'figure'),
    Input('production-company-dropdown', 'value')
)
def update_bar_chart(selected_companies):
    # Ensure that selected_companies is always a list
    if not isinstance(selected_companies, list):
        selected_companies = [selected_companies] if selected_companies else [
            movie_counts['production_companies'].iloc[0]]

    filtered_data = movie_counts[movie_counts['production_companies'].isin(selected_companies)]

    # Create a bar chart
    fig = px.bar(filtered_data, x='production_companies', y='movie_count', title='Total Movies by Production Company',color='production_companies')
    fig.update_layout(
        showlegend=True,
        title={
            'text': 'Total Number of Movies by Production Company',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'darkblue', 'size': 20}
        },
        xaxis_title='Production Company',
        yaxis_title='Number of Movies',
        xaxis=dict(
            titlefont=dict(size=18, color='red'),
        ),
        yaxis=dict(
            titlefont=dict(size=18, color='red'),
        )
    )
    return fig

## Defining callback for updating Tab 2 content

@app.callback(
    Output('revenue-bar-chart', 'figure'),
    Input('production-company-dropdown-tab2', 'value')
)
def update_revenue_chart(selected_companies):
    # Ensure that selected_companies is always a list
    if not isinstance(selected_companies, list):
        selected_companies = [selected_companies] if selected_companies else [
            revenue_by_company['production_companies'].iloc[0]]

    filtered_data = revenue_by_company[revenue_by_company['production_companies'].isin(selected_companies)]

    # Create a bar chart
    fig = px.bar(filtered_data, x='production_companies', y='total_revenue', title='Total Revenue by Production Company',color='production_companies')
    fig.update_layout(
        showlegend=True,
        title={
            'text': 'Total Revenue by Production Company',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'darkblue', 'size': 20}
        },
        xaxis_title='Production Company',
        yaxis_title='Total Revenue',
        xaxis=dict(
            titlefont=dict(size=18, color='red'),
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            titlefont=dict(size=18, color='red'),
        )
    )
    return fig

## Defining callback for updating Tab 3 content

@app.callback(
    Output('revenue-chart', 'figure'),
    [Input('month-checklist', 'value')]
)
def update_line_chart(selected_months):
    if not selected_months:
        return px.line()

    filtered_data = monthly_revenue[monthly_revenue['release_month_name'].isin(selected_months)]

    fig = px.line(filtered_data, x='release_month_name', y='revenue', title='Total Revenue by Month')
    fig.update_layout(
        title={
            'text': 'Total Revenue by Month',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'darkblue', 'size': 20}
        },
        xaxis_title='Month Names',
        yaxis_title='Total Revenue',
        xaxis=dict(
            titlefont=dict(size=18, color='red'),
        ),
        yaxis=dict(
            titlefont=dict(size=18, color='red'),
        )
    )
    return fig


# Callback for updating the revenue line chart for tab 4
@app.callback(
    Output('revenue-by-year-chart', 'figure'),
    Input('year-range-slider', 'value')
)
def update_yearly_revenue_chart(selected_year_range):
    if not selected_year_range:
        return px.line()

    # Filtering data based on the selected year range
    mask = (movies_df['year'] >= selected_year_range[0]) & (movies_df['year'] <= selected_year_range[1])
    filtered_data = movies_df[mask].groupby('year')['revenue'].sum().reset_index()

    # Create the line chart
    fig = px.line(filtered_data, x='year', y='revenue', title='Total Revenue by Year')
    fig.update_layout(
        title={
            'text': 'Total Revenue by Year',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'darkblue', 'size': 20}
        },
        xaxis_title='Year',
        yaxis_title='Total Revenue',
        xaxis=dict(
            titlefont=dict(size=18, color='red'),
        ),
        yaxis=dict(
            titlefont=dict(size=18, color='red'),
        )
    )
    return fig

## Callback for Tab 5

@app.callback(
    Output('status-pie-chart', 'figure'),
    Input('status-checklist', 'value')
)
def update_status_pie_chart(selected_statuses):
    if not selected_statuses:
        return px.pie()  # Returns an empty pie chart if no statuses are selected

    # Filter data based on selected statuses
    filtered_data = movies_df[movies_df['status'].isin(selected_statuses)]
    status_counts = filtered_data['status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']

    # Create a pie chart using Plotly Express
    fig = px.pie(status_counts, values='count', names='status', title='Total Number of Movies by Status')
    fig.update_traces(textposition='outside', textinfo='percent+label+value')
    fig.update_layout(
        width = 800,
        height = 600,
        margin=dict(t=350, l=0, r=0, b=0),  # Adjust top margin to make space for title
        title={
            'text': 'Total Number of Movies by Statuses',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'darkblue', 'size': 20}
        },
        # If needed, adjust the legend position
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    return fig


# ===============Phase 5 -  Running the app ================
if __name__ == '__main__':
    app.run_server(debug=False,
                   port=8030,
                   host='0.0.0.0')