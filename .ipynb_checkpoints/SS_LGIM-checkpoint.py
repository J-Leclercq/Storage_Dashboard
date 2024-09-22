#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# Step 1: Load and Prepare Data
merged_data = pd.read_csv('merged_data.csv')
population_data = pd.read_csv('population.csv')

# Clean column names and relevant data
merged_data.columns = merged_data.columns.str.strip()
population_data.columns = population_data.columns.str.strip()
merged_data['Local Authority'] = merged_data['Local Authority'].str.strip().str.lower()
population_data['local authorities'] = population_data['local authorities'].str.strip().str.lower()
merged_data['Typical Floor Size'] = pd.to_numeric(merged_data['Typical Floor Size'], errors='coerce')

# Remove duplicates and aggregate
unique_properties = merged_data.drop_duplicates(subset=['Property Address', 'Local Authority'])
aggregated_floor_size = unique_properties.groupby('Local Authority', as_index=False)['Typical Floor Size'].sum()

# Load GeoDataFrame and merge with population data
geojson_file = 'LAs.geojson'
gdf = gpd.read_file(geojson_file)
gdf['LAD21NM'] = gdf['LAD21NM'].str.strip().str.lower()
gdf = gdf.merge(population_data[['local authorities', 'Total Population']], left_on='LAD21NM', right_on='local authorities', how='left')
gdf = gdf.merge(aggregated_floor_size, left_on='LAD21NM', right_on='Local Authority', how='left')

# Convert and calculate necessary columns
gdf['Total Square Foot'] = pd.to_numeric(gdf['Typical Floor Size'], errors='coerce').fillna(0)
gdf['Total Population'] = pd.to_numeric(gdf['Total Population'], errors='coerce').fillna(0)
gdf['SqFt_Per_Capita'] = gdf['Total Square Foot'] / gdf['Total Population']
gdf['SqFt_Per_Capita'] = gdf['SqFt_Per_Capita'].replace([float('inf'), -float('inf')], 0).fillna(0)  # Replace NaNs with 0

# Set areas with missing or zero SqFt_Per_Capita to None for greying out in the map
gdf['SqFt_Per_Capita_Map'] = gdf['SqFt_Per_Capita'].apply(lambda x: None if x == 0 else x)

# Step 2: Create the Choropleth Map with Correct GeoJSON
map_chart = px.choropleth_mapbox(
    gdf,
    geojson=gdf.__geo_interface__,  # Use the extracted GeoJSON data
    locations='LAD21NM',  # Match locations by local authority names
    featureidkey="properties.LAD21NM",  # Match features in GeoJSON with LAD21NM
    color='SqFt_Per_Capita_Map',
    hover_name='LAD21NM',  # Display Local Authority Names
    hover_data={
        'SqFt_Per_Capita': ':.2f',  # Format Sq Ft Per Capita to 2 decimal places
        'Total Population': ':.0f',
        'Total Square Foot': ':.0f',
        'SqFt_Per_Capita_Map': False,  # Exclude the extra map field from the tooltip
        'LAD21NM': False,  # Hide the raw LAD21NM field
    },
    labels={'LAD21NM': 'Local Authority', 'SqFt_Per_Capita': 'Sq Ft Per Capita'},
    mapbox_style="carto-positron",
    center={"lat": 54.0, "lon": -2.0},
    zoom=5,
    title='Sq Ft Per Capita Heatmap',
    color_continuous_scale="YlGnBu",  # Change to a subtle yellow-to-blue gradient
    range_color=(0.1, gdf['SqFt_Per_Capita_Map'].max()),  # Ensure the range starts above 0 for proper display
)

# Step 3: Create the Bubble Chart
bubble_chart = px.scatter(
    gdf,
    x='Total Population',
    y='Total Square Foot',
    size='SqFt_Per_Capita',
    color='SqFt_Per_Capita',
    hover_name='LAD21NM',
    title='Sq Ft Per Capita Bubble Chart',
    labels={
        'Total Population': 'Total Population',
        'Total Square Foot': 'Total Square Foot (Total Space)',
        'SqFt_Per_Capita': 'Sq Ft Per Capita'
    },
    size_max=60,
    hover_data={
        'Total Population': ':.0f',
        'Total Square Foot': ':.0f',
        'SqFt_Per_Capita': ':.2f'
    }
)

# Step 4: Create the Interactive Table
table_data = gdf[['LAD21NM', 'Total Square Foot', 'Total Population', 'SqFt_Per_Capita']].copy()
total_sq_ft = table_data['Total Square Foot'].sum()
total_population = table_data['Total Population'].sum()
avg_sqft_per_capita = total_sq_ft / total_population if total_population > 0 else 0
summary_row = pd.DataFrame([{
    'LAD21NM': 'Total',
    'Total Square Foot': total_sq_ft,
    'Total Population': total_population,
    'SqFt_Per_Capita': avg_sqft_per_capita
}])
table_data = pd.concat([summary_row, table_data.sort_values(by='SqFt_Per_Capita', ascending=False)], ignore_index=True)

# Step 5: Create the Top 10 Owners Bar Chart
owner_aggregation = merged_data.groupby('Owner Name', as_index=False)['Typical Floor Size'].sum()
owner_aggregation = owner_aggregation.rename(columns={'Typical Floor Size': 'Total Square Foot'})
top_10_owners = owner_aggregation.nlargest(10, 'Total Square Foot')

owner_chart = px.bar(
    top_10_owners,
    x='Owner Name',
    y='Total Square Foot',
    title='Top 10 Owners by Total Square Foot',
    labels={'Total Square Foot': 'Total Square Foot (SF)', 'Owner Name': 'Owner'},
    text='Total Square Foot'
)
owner_chart.update_traces(texttemplate='%{text:.2s}', textposition='outside')
owner_chart.update_layout(
    height=600,
    margin=dict(l=0, r=0, t=40, b=0)
)

# Streamlit Layout
st.title("Local Authority Analysis Dashboard")

# Display the maps and charts
st.plotly_chart(map_chart, use_container_width=True)
st.plotly_chart(bubble_chart, use_container_width=True)
st.write("### Local Authority Space Analysis Table")
st.dataframe(table_data)
st.plotly_chart(owner_chart, use_container_width=True)


# In[5]:


jupyter nbconvert --to script SS_LGIM.ipynb


# In[ ]:




