import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
    
def add_year(dfs):
    for year in dfs:
        dfs[year]['Year'] = int(year)

def load_data():
    # Load the data
    dfs = { '2015' : pd.read_csv('data/2015.csv'),
            '2016' : pd.read_csv('data/2016.csv'),
            '2017' : pd.read_csv('data/2017.csv'),
            '2018' : pd.read_csv('data/2018.csv'),
            '2019' : pd.read_csv('data/2019.csv')}

    #Standardise column names
    dfs['2017'].columns = ['Country', 'Happiness Rank', 'Happiness Score', 
                    'Upper Confidence Interval', 'Lower Confidence Interval', 
                    'Economy (GDP per Capita)', 'Family', 
                    'Health (Life Expectancy)', 'Freedom', 
                    'Trust (Government Corruption)', 'Generosity', 
                    'Dystopia Residual']
    dfs['2018'].columns = ['Happiness Rank', 'Country', 'Happiness Score', 
                    'Economy (GDP per Capita)', 'Family', 
                    'Health (Life Expectancy)', 'Freedom', 'Generosity', 
                    'Trust (Government Corruption)']
    dfs['2019'].columns = ['Happiness Rank', 'Country', 'Happiness Score', 
                    'Economy (GDP per Capita)', 'Family', 
                    'Health (Life Expectancy)', 'Freedom', 'Generosity', 
                    'Trust (Government Corruption)'] 

    return dfs

def add_metrics(metrics, df_mean, df_baseline_mean=None):
    st.write('**Mean Metric Values with Year on Year Change**')
    
    col1, col2, col3, col4 = st.columns(4)
    columns = [col1, col2, col3, col4]
    count = 0
    for met in metrics:
        with columns[count % 4]:
            if df_baseline_mean is not None:
                delta = (df_mean[met] - df_baseline_mean[met]) / \
                    df_baseline_mean[met]
                st.metric(label=met, value=round(df_mean[met], 2), delta="{:.2%}".format(delta))
            else:
                st.metric(label=met, value=round(df_mean[met], 2))
            count += 1

def add_cor_matrix(df):
    df_corr = df.drop(columns=['Country', 'Region', 'Happiness Rank', 
        'Standard Error', 'Year', 'Lower Confidence Interval',
       'Upper Confidence Interval', 'Country Code'], errors='ignore')
    spearman_cor_matrix = df_corr.corr(method='spearman')
    fig = px.imshow(spearman_cor_matrix, text_auto='.2f', 
                    zmin=-1, zmax=1, 
                    color_continuous_scale='viridis',
                    title='Spearman Correlation between Metrics')
    fig.update_xaxes(side='top')
    fig.update_layout(height=800)
    st.plotly_chart(fig, height=800)
    
def add_regional_analysis(df):
    fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Mean Happiness Score", "Happiness Rank Distribution"))
    df_regional_avg = df.groupby(['Region']).mean(numeric_only=True)
    fig.add_trace(go.Bar(x=df_regional_avg.index, y=df_regional_avg['Happiness Score']), row=1, col=1)
    fig.add_trace(go.Box(x=df['Region'], y=df['Happiness Rank']), row=1, col=2)
    fig.update_layout(showlegend=False, title='Regional Happiness Analyisis')
    st.plotly_chart(fig)

def add_happiness_world_map(df):

    fig = px.choropleth(df, locations="Country",
                        locationmode='country names',
                    color="Happiness Score",
                    hover_name="Country", 
                    color_continuous_scale=px.colors.sequential.Plasma_r,
                    title='Happiness Score World Map')
    fig.update_layout(height=800)
    st.plotly_chart(fig, height=800)

def add_metric_delta_world_map(df, metric_names):
    map_selector = st.selectbox(label='Happiness Metric', options=metric_names)
    fig = px.choropleth(df, locations="Country",
                        locationmode='country names',
                    color=map_selector,
                    hover_name="Country", 
                    color_continuous_scale=px.colors.sequential.Plasma_r,
                    title=f'Change in {map_selector} World Map')
    fig.update_layout(height=800)
    st.plotly_chart(fig, height=800)


def create_year_screen(dfs, mean_dfs, year):
    # Define all possible desired metrics
    metric_names = ['Happiness Score', 'Economy (GDP per Capita)', 'Family', 
                 'Health (Life Expectancy)', 'Freedom', 
                 'Trust (Government Corruption)', 'Generosity', 
                 'Dystopia Residual']
    
    # Find the metrics available in the given years data
    metric_names_in_df = set(metric_names) & set(mean_dfs[str(year)].index)

    # Create the world map
    add_happiness_world_map(dfs[str(year)])

    
    # Perform a regional analysis if regions are named
    if 'Region' in dfs[str(year)].columns:
        add_regional_analysis(dfs[str(year)])

    # Create metrics with comparison if comparison data exists
    if str(year-1) in mean_dfs:
        metric_names_in_df = set(metric_names_in_df) & set(mean_dfs[str(year-1)].index)
        
        add_metrics(metric_names_in_df, mean_dfs[str(year)], mean_dfs[str(year-1)])
    else:
        add_metrics(metric_names_in_df, mean_dfs[str(year)])

    # Create the spearman correlation matrix
    add_cor_matrix(dfs[str(year)])


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # Load the data 
    dfs = load_data()
    
    # Add year data
    add_year(dfs)
    
    # Create a combined dataframe
    df_all = pd.concat(dfs.values(), 
                       ignore_index=True)
    
    # Find mean values from the datasets
    mean_dfs = {'2015' : dfs['2015'].mean(numeric_only=True),
                '2016' : dfs['2016'].mean(numeric_only=True),
                '2017' : dfs['2017'].mean(numeric_only=True),
                '2018' : dfs['2018'].mean(numeric_only=True),
                '2019' : dfs['2019'].mean(numeric_only=True)}
    
    # Find delta values for delta world map
    # Define the possible metrics to compare 
    metric_names = ['Happiness Score', 'Economy (GDP per Capita)', 'Family', 
                 'Health (Life Expectancy)', 'Freedom', 
                 'Trust (Government Corruption)', 'Generosity', 
                 'Dystopia Residual']
    # Find the first and latest year available
    first_year, final_year = str(min(list(map(int, dfs.keys())))), str(max(list(map(int, dfs.keys()))))
    # Filter the metric names to the ones present in both dataframes
    delta_metric_names = set(metric_names) & set(dfs[first_year].columns) & set(dfs[final_year].columns)
    # Create a dataframe of deltas
    first_year_df, final_year_df = dfs[first_year].set_index('Country'), dfs[final_year].set_index('Country')
    delta_df = final_year_df.sub(first_year_df)
    # Filter the data frame to only include the desired metrics
    delta_df = delta_df[list(delta_metric_names)]
    delta_df.reset_index(names=['Country'], inplace=True)
    
    # Summary

    # Regression comparisons
    # Multiple Linear, SVR, Decision tree, Random Forest
    
    # Build the dashboard
    add_sidebar = st.sidebar.selectbox('Year', ('Summary', '2019', '2018', '2017', '2016', '2015'))
    

    if add_sidebar == 'Summary':
        #Create a line chart of the top 5 happiest countries
        fig = px.line(df_all[df_all['Happiness Rank'] <= 5], x='Year', y='Happiness Rank', color='Country', title="Top 5 Ranked Nations")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig)

        # Create a world map showing the change of each metric from start to end
        add_metric_delta_world_map(delta_df, delta_metric_names)
        
    if add_sidebar == '2019':
        create_year_screen(dfs, mean_dfs, 2019)

    if add_sidebar == '2018':
        create_year_screen(dfs, mean_dfs, 2018)

    if add_sidebar == '2017':
        create_year_screen(dfs, mean_dfs, 2017)

    if add_sidebar == '2016':
        create_year_screen(dfs, mean_dfs, 2016)
        
    if add_sidebar == '2015':
        create_year_screen(dfs, mean_dfs, 2015)
