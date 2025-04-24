import pandas as pd
import numpy as np
import pycountry
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

def get_confidence_interval_from_standard_error(df):
    #Calculate a 95% confidence interval
    df['Lower Confidence Interval'] = df['Happiness Score'] - \
        1.96*df['Standard Error']
    df['Upper Confidence Interval'] = df['Happiness Score'] + \
        1.96*df['Standard Error']

def get_standard_error_from_confidence_interval(df):
    #Calculate a standard error assuming a 95% confidence interval
    df['Standard Error'] = df['Upper Confidence Interval'] - \
        df['Lower Confidence Interval'] / 3.92
    
def add_year_and_code(dfs, years):
    for df, year in zip(dfs, years):
        df['Year'] = year
        df['Country Code'] = df['Country']
        for index, row in df.iterrows():
            try:
                country = pycountry.countries.search_fuzzy(row['Country Code'])
                df.at[index, 'Country Code'] = country[0].alpha_3
            except:
                df.at[index, 'Country Code'] = np.nan

def load_data():
    # Load the data
    df_2015 = pd.read_csv('data/2015.csv')
    df_2016 = pd.read_csv('data/2016.csv')
    df_2017 = pd.read_csv('data/2017.csv')
    df_2018 = pd.read_csv('data/2018.csv')
    df_2019 = pd.read_csv('data/2019.csv')

    #Standardise column names
    df_2017.columns = ['Country', 'Happiness Rank', 'Happiness Score', 
                    'Upper Confidence Interval', 'Lower Confidence Interval', 
                    'Economy (GDP per Capita)', 'Family', 
                    'Health (Life Expectancy)', 'Freedom', 
                    'Trust (Government Corruption)', 'Generosity', 
                    'Dystopia Residual']
    df_2018.columns = ['Happiness Rank', 'Country', 'Happiness Score', 
                    'Economy (GDP per Capita)', 'Family', 
                    'Health (Life Expectancy)', 'Freedom', 'Generosity', 
                    'Trust (Government Corruption)']
    df_2019.columns = ['Happiness Rank', 'Country', 'Happiness Score', 
                    'Economy (GDP per Capita)', 'Family', 
                    'Health (Life Expectancy)', 'Freedom', 'Generosity', 
                    'Trust (Government Corruption)'] 

    return df_2015, df_2016, df_2017, df_2018, df_2019

def add_metrics(metrics, df_mean, df_baseline_mean=None):
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
    fig = px.imshow(spearman_cor_matrix, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='viridis')
    fig.update_xaxes(side='top')
    st.plotly_chart(fig)
    
def add_regional_analysis(df):
    fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Regional Score", "Regional Rank"))
    df_regional_avg = df.groupby(['Region']).mean(numeric_only=True)
    fig.add_trace(go.Bar(x=df_regional_avg.index, y=df_regional_avg['Happiness Score']), row=1, col=1)
    fig.add_trace(go.Box(x=df['Region'], y=df['Happiness Rank']), row=1, col=2)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

def add_world_map(df):
    fig = px.choropleth(df, locations="Country Code",
                    color="Happiness Score", # lifeExp is a column of gapminder
                    hover_name="Country", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(fig)

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # Load the data 
    df_2015, df_2016, df_2017, df_2018, df_2019 = load_data()
    # Add year data
    add_year_and_code(dfs=[df_2015, df_2016, df_2017, df_2018, df_2019],
             years=[2015, 2016, 2017, 2018, 2019])
    # Create a combined dataframe
    df_all = pd.concat([df_2015, df_2016, df_2017, df_2018, df_2019], 
                       ignore_index=True)
    

    # Perform data engineering
    # Calculate missing confidence interval and standard error data
    get_confidence_interval_from_standard_error(df_2015)
    get_standard_error_from_confidence_interval(df_2016)
    get_standard_error_from_confidence_interval(df_2017)

    # Analysis to perform
    
    # Summary

    # Regression comparisons
    # Multiple Linear, SVR, Decision tree, Random Forest
    


    # Yearly

    # 3D scatter plot
    # Happiness on z also shown by colour with 2 variables
    # on the x and y 

    # World Heat Map
    # Colour country based on score

    # print(df_2015.columns)
    # print(df_2016.columns)
    # print(df_2017.columns)
    # print(df_2018.columns)
    # print(df_2019.columns)
    # print(df_all.columns)


    
    # Build the dashboard
    add_sidebar = st.sidebar.selectbox('Year', ('Summary', '2019', '2018', '2017', '2016', '2015'))
    mean_2015 = df_2015.mean(numeric_only=True)
    mean_2016 = df_2016.mean(numeric_only=True)
    mean_2017 = df_2017.mean(numeric_only=True)
    mean_2018 = df_2018.mean(numeric_only=True)
    mean_2019 = df_2019.mean(numeric_only=True)

    if add_sidebar == 'Summary':
        fig = px.line(df_all[df_all['Happiness Rank'] <= 5], x='Year', y='Happiness Rank', color='Country', title="Top 5 Ranked Nations")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig)
        # TODO
        st.write("TODO")
    if add_sidebar == '2019':
        add_metrics(['Happiness Score', 'Economy (GDP per Capita)', 'Family', 
                     'Health (Life Expectancy)', 'Freedom', 
                     'Trust (Government Corruption)', 'Generosity'], 
                     mean_2019, mean_2018)
        add_world_map(df_2019)
        add_cor_matrix(df_2019)

        # TODO
        st.write("TODO")
    if add_sidebar == '2018':
        add_metrics(['Happiness Score', 'Economy (GDP per Capita)', 'Family', 
                     'Health (Life Expectancy)', 'Freedom', 
                     'Trust (Government Corruption)', 'Generosity'], 
                     mean_2018, mean_2017)
        add_world_map(df_2018)
        add_cor_matrix(df_2018)
        # TODO
        st.write("TODO")
    if add_sidebar == '2017':
        add_metrics(['Happiness Score', 'Economy (GDP per Capita)', 'Family', 
                     'Health (Life Expectancy)', 'Freedom', 
                     'Trust (Government Corruption)', 'Generosity', 
                     'Dystopia Residual'], mean_2017, mean_2016)
        add_world_map(df_2017)
        add_cor_matrix(df_2017)
        # TODO
        st.write("TODO")
    if add_sidebar == '2016':
        add_metrics(['Happiness Score', 'Economy (GDP per Capita)', 'Family', 
                     'Health (Life Expectancy)', 'Freedom', 
                     'Trust (Government Corruption)', 'Generosity', 
                     'Dystopia Residual'], mean_2016, mean_2015)
        add_world_map(df_2016)
        add_cor_matrix(df_2016)
        add_regional_analysis(df_2016)
        # TODO
        st.write("TODO")
    if add_sidebar == '2015':
        add_metrics(['Happiness Score', 'Economy (GDP per Capita)', 'Family', 
                     'Health (Life Expectancy)', 'Freedom', 
                     'Trust (Government Corruption)', 'Generosity', 
                     'Dystopia Residual'], mean_2015)
        add_world_map(df_2015)
        add_cor_matrix(df_2015)
        add_regional_analysis(df_2015)
        # TODO
        st.write("TODO")

# '''
# 2015 : ['Country', 'Region', 'Happiness Rank', 'Happiness Score',
#        'Standard Error', 'Economy (GDP per Capita)', 'Family',
#        'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
#        'Generosity', 'Dystopia Residual', 'Year', 'Lower Confidence Interval',
#        'Upper Confidence Interval'],
#       dtype='object')
# 2016 : (['Country', 'Region', 'Happiness Rank', 'Happiness Score',
#        'Lower Confidence Interval', 'Upper Confidence Interval',
#        'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
#        'Freedom', 'Trust (Government Corruption)', 'Generosity',
#        'Dystopia Residual', 'Year', 'Standard Error'],
#       dtype='object')
# 2017 : (['Country', 'Happiness Rank', 'Happiness Score',
#        'Upper Confidence Interval', 'Lower Confidence Interval',
#        'Economy (GDP per Capita)', 'Family', 'Health (Life Expectancy)',
#        'Freedom', 'Trust (Government Corruption)', 'Generosity',
#        'Dystopia Residual', 'Year', 'Standard Error'],
#       dtype='object')
# 2018 : (['Happiness Rank', 'Country', 'Happiness Score',
#        'Economy (GDP per Capita)', 'Social support',
#        'Health (Life Expectancy)', 'Freedom', 'Generosity',
#        'Trust (Government Corruption)', 'Year'],
#       dtype='object')
# 2019 : (['Happiness Rank', 'Country', 'Happiness Score',
#        'Economy (GDP per Capita)', 'Social support',
#        'Health (Life Expectancy)', 'Freedom', 'Generosity',
#        'Trust (Government Corruption)', 'Year'],
#       dtype='object')
# ALL : (['Country', 'Region', 'Happiness Rank', 'Happiness Score',
#        'Standard Error', 'Economy (GDP per Capita)', 'Family',
#        'Health (Life Expectancy)', 'Freedom', 'Trust (Government Corruption)',
#        'Generosity', 'Dystopia Residual', 'Year', 'Lower Confidence Interval',
#        'Upper Confidence Interval', 'Social support'],
# '''
