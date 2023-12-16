import pandas as pd
import sklearn
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString
from sklearn.neighbors import BallTree
import pyproj
import plotly.express as px
import plotly.graph_objects as go
from shapely.ops import nearest_points
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
from itertools import product
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px

print('maindf importing')
maindf = pd.read_csv('./substationsPowesAlti.csv')
print('df_wind importing')
df_wind = pd.read_csv('./testtt.csv')
# df_wind = pd.read_csv('../../Downloads/tensorflow-test/testtt.csv')
print('df_power_curves importing')
df_power_curves = pd.read_csv("./power_curves.csv")
print('redefine_aggregation defining')
df_wind = pd.merge(df_wind, maindf, on=['Latitude', 'Longitude'], how='left')
print('merging')
def redefine_aggregation(df_wind, weight_factors, hourly_energy_production_turbine_name):
    print('using hourly_energy_production_turbine_name', hourly_energy_production_turbine_name)
    # First, fill missing wind gust values with the wind speed values
    df_wind['wind_gust'].fillna(df_wind['wind_speed'], inplace=True)
    
    # Sum up the hourly energy production to get the total annual energy production for each location
    df_annual_energy = df_wind.groupby(['Latitude', 'Longitude'])[hourly_energy_production_turbine_name].sum().reset_index()
    df_annual_energy.rename(columns={hourly_energy_production_turbine_name: 'annual_energy_production_kWh'}, inplace=True)
    
    # Normalize the annual energy production
    max_energy = df_annual_energy['annual_energy_production_kWh'].max()
    min_energy = df_annual_energy['annual_energy_production_kWh'].min()
    df_annual_energy['normalized_energy_production'] = (df_annual_energy['annual_energy_production_kWh'] - min_energy) / (max_energy - min_energy)
    
    # Calculate infrastructure cost factor and normalize it
    df_wind['infrastructure_cost_factor'] = 1 / (df_wind['distance_to_nearest_powerline_m'] + df_wind['distance_to_nearest_substation_m'])
    max_infra_cost = df_wind['infrastructure_cost_factor'].max()
    min_infra_cost = df_wind['infrastructure_cost_factor'].min()
    df_wind['normalized_infrastructure_cost'] = (df_wind['infrastructure_cost_factor'] - min_infra_cost) / (max_infra_cost - min_infra_cost)
    
    # Calculate altitude penalty
    df_wind['altitude_penalty'] = df_wind['altitude'].apply(lambda x: 0 if x >= 0 else abs(x)**weight_factors['altitude_penalty'])
    
    # Merge the normalized annual energy production back into the main DataFrame
    df_wind = df_wind.merge(df_annual_energy[['Latitude', 'Longitude', 'normalized_energy_production']], on=['Latitude', 'Longitude'])
    
    # Calculate profitability
    df_wind['profitability'] = (
        weight_factors['energy_production'] * df_wind['normalized_energy_production'] -
        weight_factors['infrastructure_cost'] * df_wind['normalized_infrastructure_cost'] -
        weight_factors['environmental_factor'] * df_wind['altitude_penalty']
    )
    
    # Group by latitude and longitude to aggregate the results
    df_aggregated = df_wind.groupby(['Latitude', 'Longitude']).agg({
        'normalized_energy_production': 'first',
        'normalized_infrastructure_cost': 'first',
        'altitude_penalty': 'first',
        'profitability': 'mean'  # Using mean to represent the average profitability over time
    }).reset_index()
    
    return df_aggregated
print('plot_with_px defining')
def plot_with_px(lon, lat, current_zoom):
    df_random_point = pd.DataFrame({'Longitude': [lon], 'Latitude': [lat]})
    gdf_random_point = gpd.GeoDataFrame(df_random_point, geometry=gpd.points_from_xy(df_random_point.Longitude, df_random_point.Latitude))
    gdf_random_point.crs = 'EPSG:4326'

    substations = gpd.read_file('../ex-substations.geojson')
    powerlines = gpd.read_file('../existing-transmission-lines.geojson')
    substations.crs = powerlines.crs = 'EPSG:4326'

    utm_zone = pyproj.CRS.from_epsg(32634)
    gdf_random_point_utm = gdf_random_point.to_crs(utm_zone)
    substations_utm = substations.to_crs(utm_zone)
    powerlines_utm = powerlines.to_crs(utm_zone)

    tree = BallTree(np.array(list(zip(substations_utm.geometry.x, substations_utm.geometry.y))))
    distance_substation, index_substation = tree.query(np.array(list(zip(gdf_random_point_utm.geometry.x, gdf_random_point_utm.geometry.y))), k=1)
    nearest_substation_line = LineString([gdf_random_point_utm.geometry.iloc[0], substations_utm.geometry.iloc[index_substation[0][0]]])
    dist_to_substation = distance_substation[0][0]  # Distance to nearest substation in meters

    min_dist = np.inf
    nearest_point_on_line = None
    for line in powerlines_utm.geometry:
        np1, np2 = nearest_points(gdf_random_point_utm.geometry.iloc[0], line)
        dist = gdf_random_point_utm.geometry.iloc[0].distance(np2)
        if dist < min_dist:
            min_dist = dist
            nearest_point_on_line = np2
    nearest_powerline_line = LineString([gdf_random_point_utm.geometry.iloc[0], nearest_point_on_line])
    dist_to_powerline = min_dist  # Distance to nearest powerline in meters

    nearest_substation_line_gdf = gpd.GeoDataFrame(geometry=[nearest_substation_line], crs=utm_zone)
    nearest_powerline_line_gdf = gpd.GeoDataFrame(geometry=[nearest_powerline_line], crs=utm_zone)

    nearest_substation_line_wgs84 = nearest_substation_line_gdf.to_crs('EPSG:4326')
    nearest_powerline_line_wgs84 = nearest_powerline_line_gdf.to_crs('EPSG:4326')

    fig = px.scatter_mapbox(substations, lat=substations.geometry.y, lon=substations.geometry.x, 
                            color_discrete_sequence=['red'], zoom=current_zoom, height=1000)

    fig.add_trace(go.Scattermapbox(
        lat=[df_random_point['Latitude'][0]],
        lon=[df_random_point['Longitude'][0]],
        mode='markers',
        marker=go.scattermapbox.Marker(size=9, color='blue'),
        name='Point'
    ))

    # Plot power lines
    for _, row in powerlines.iterrows():
        if row.geometry.geom_type == 'LineString':
            lon, lat = row.geometry.xy
            lon_list = lon.tolist()
            lat_list = lat.tolist()
            fig.add_trace(go.Scattermapbox(
                lon=lon_list,
                lat=lat_list,
                mode='lines',
                line=go.scattermapbox.Line(color='green', width=2),
                showlegend=False
            ))
        elif row.geometry.geom_type == 'MultiLineString':
            for geom in row.geometry:
                lon, lat = geom.xy
                lon_list = lon.tolist()
                lat_list = lat.tolist()
                fig.add_trace(go.Scattermapbox(
                    lon=lon_list,
                    lat=lat_list,
                    mode='lines',
                    line=go.scattermapbox.Line(color='green', width=2),
                    showlegend=False
                ))

    def add_line_with_label(fig, line, label, color, distance):
        # Extract the coordinates of the line
        lon, lat = line.geometry.iloc[0].xy
        lon_list = lon.tolist()
        lat_list = lat.tolist()

        # Create the hover text with the rounded distance
        hover_text = f"{label}: {np.round(distance, 2)} m"

        # Add the line to the figure with hover information
        fig.add_trace(go.Scattermapbox(
            lon=lon_list,
            lat=lat_list,
            mode='lines',
            line=go.scattermapbox.Line(color=color, width=2),
            name=label,
            hoverinfo='text',
            hovertext=hover_text  # This is the text that will appear on hover
        ))


    # Use the modified function in your main plotting function
    add_line_with_label(fig, nearest_substation_line_wgs84, 'Distance to Substation', 'purple', dist_to_substation)
    add_line_with_label(fig, nearest_powerline_line_wgs84, 'Distance to Power Line', 'orange', dist_to_powerline)
    fig.update_layout(mapbox_style="open-street-map")

    return fig
print('create_power_curves_graph defining')
def create_power_curves_graph(df_power_curves):
    fig = go.Figure()
    for turbine in df_power_curves.columns[1:]:  # Assuming first column is Wind Speed
        fig.add_trace(go.Scatter(x=df_power_curves['Wind Speed'], y=df_power_curves[turbine],
                                 mode='lines', name=turbine))
    
    fig.update_layout(
        title='Power Curves',
        xaxis_title='Wind Speed (m/s)',
        yaxis_title='Power (KW)',
        legend=dict(
            orientation='h',
            x=0.5,  # Centers the legend horizontally
            y=-0.2,  # Positions the legend below the x-axis
            xanchor='center',
            yanchor='top'
        )
    )
    return fig
print('highlight_profitable_locations defining')
def highlight_profitable_locations(fig, df, threshold):
    high_profit_locations = df[df['profitability'] >= threshold]
    for index, row in high_profit_locations.iterrows():
        fig.add_trace(
            go.Scattermapbox(
                lon=[row['Longitude']],
                lat=[row['Latitude']],
                mode='markers',
                marker=go.scattermapbox.Marker(
                    size=15,
                    color='red',
                    opacity=0.5
                ),
                hoverinfo='skip'
            )
        )
    return fig
print('running the code')

weight_factors = {
    'energy_production': 0.9,  # Adjust as needed
    'infrastructure_cost': 0.07,  # Adjust as needed
    'environmental_factor': 0.03,  # Adjust as needed
    'altitude_penalty': 0.06  # Adjust as needed
}
df_aggregated = redefine_aggregation(df_wind, weight_factors, 'hourly_energy_production')
power_curves_fig = create_power_curves_graph(df_power_curves)
map_center = [df_aggregated['Latitude'].mean(), df_aggregated['Longitude'].mean()]

map_center = [df_aggregated['Latitude'].mean(), df_aggregated['Longitude'].mean()]
df_aggregated['size_for_plot'] = df_aggregated['profitability'].abs()
initial_fig = px.scatter_mapbox(df_aggregated, lat='Latitude', lon='Longitude',
                                color='profitability', size='size_for_plot', 
                                size_max=15, zoom=6,
                                center=dict(lat=map_center[0], lon=map_center[1]),
                                mapbox_style="open-street-map",
                                opacity=0.8)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
global rendered_fig
rendered_fig = initial_fig

# Define the sidebar style for full-screen height
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}
FIGURE_STYLE = {'height': '92vh'}

app.layout = html.Div([
    dbc.Row(
        dbc.Col(html.H1("Albanian Wind Data Study", className="text-center"), width=12)
    ),
    dbc.Row([
        dbc.Col(
            html.Div(
                [
                    html.H5("Filters", className="display-7"),
                    html.Hr(),
                    html.P("Set the parameters for the wind study", className="lead"),
                    dbc.Label("Wind Speed Weight:"),
                    dbc.Input(id='input-wind-speed', type='number', placeholder="Wind Speed Weight", value=0.9, className="mb-3"),
                    dbc.Label("Infrastructure Cost Weight:"),
                    dbc.Input(id='input-infrastructure-cost', type='number', placeholder="Infrastructure Cost Weight", value=0.07, className="mb-3"),
                    dbc.Label("Environmental Factor Weight:"),
                    dbc.Input(id='input-environmental-factor', type='number', placeholder="Environmental Factor Weight", value=0.03, className="mb-3"),
                    dbc.Label("Altitude Penalty"),
                    dbc.Input(id='input-altitude-penalty', type='number', placeholder="Altitude Penalty", value=0.06, className="mb-3"),
                    dbc.Label("Turbine Name:"),
                    dcc.Dropdown(
                        id='dropdown-menu',
                        options=[
                            {"label": "GE Energy 3.6sl - 3.6MW", "value": "hourly_energy_production_GE_3_6sl"},
                            {"label": "GE Energy 2.5xl - 2.5MW", "value": "hourly_energy_production_GE_2_5xl"},
                            {"label": "GE Energy 1.5s - 1.5MW", "value": "hourly_energy_production_GE_1_5s"},
                            {"label": "Siemens-Gamesa 8.0-167 DD - 8.0MW", "value": "hourly_energy_production"}
                        ],
                        value='hourly_energy_production_GE_1_5s',
                        className="mb-3"
                    ),
                    html.Div(
                        [
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button("Recalculate", id="recalculate-button", n_clicks=0, className="mb-2"),  # mb-2 adds margin-bottom
                                    width=12
                                )
                            ),
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button("Reset", id="reset-button", n_clicks=0, className="mb-2"),  # mb-2 adds margin-bottom
                                    width=12
                                )
                            ),
                            dbc.Row(
                                dbc.Col(
                                    dbc.Button("Reveal Best Location", id="reveal-best-location-button", n_clicks=0, className="mb-2"),  # mb-2 adds margin-bottom
                                    width=12
                                )
                            )
                        ],
                        style={"padding": "0 0rem"}  # Adds padding to the left and right of the container
                    )

                ],
                style=SIDEBAR_STYLE
            ),
            width=2
        ),
        dbc.Col(dcc.Loading(
            id="loading-map",
            type="default",
            children=[dcc.Graph(
                id='map', 
                figure=initial_fig, 
                style={
                    'height': '92vh',  # Adjust the height as required
                    'background-color': 'rgb(60, 60, 60)',  # Background with 50% transparency
                    'border': '1px solid black'  # Border
                }
            )]
        ), width=7),
        dbc.Col(dcc.Graph(
            id='power-curves-graph', 
            figure=power_curves_fig, 
            style={
                'background-color': 'rgb(60, 60, 60)',  # Background with 50% transparency
                'border': '1px solid black'  # Border
            }
        ), width=3),
    ], className="g-0"),
], style={'height': FIGURE_STYLE['height']})

@app.callback(
    Output('map', 'figure'),
    [Input('reveal-best-location-button', 'n_clicks')],
    [Input('reset-button', 'n_clicks'),
     Input('recalculate-button', 'n_clicks'),
     Input('map', 'clickData')],
    [State('input-wind-speed', 'value'),
     State('input-infrastructure-cost', 'value'),
     State('input-environmental-factor', 'value'),
     State('input-altitude-penalty', 'value'),
     State('dropdown-menu', 'value'),
     State('map', 'relayoutData')]
)
def update_map(n_clicks_reveal, n_clicks_reset, n_clicks_recalculate, clickData, wind_speed_weight, infrastructure_cost_weight, environmental_factor_weight, altitude_penalty, dropdown_value, relayoutData):
# def update_map(n_clicks_reveal, n_clicks_reset, n_clicks_recalculate, current_fig, wind_speed_weight, infrastructure_cost_weight, environmental_factor_weight, altitude_penalty, dropdown_value):
    global rendered_fig  # declare rendered_fig as a global variable
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not ctx.triggered:
        rendered_fig = initial_fig
        return initial_fig

    # Define default values for clicked_lat and clicked_lon
    clicked_lat = None
    clicked_lon = None

    # Update clicked_lat and clicked_lon if there is clickData
    if clickData:
        clicked_lat = clickData['points'][0]['lat']
        clicked_lon = clickData['points'][0]['lon']

    current_zoom = relayoutData.get('mapbox.zoom') if relayoutData and 'mapbox.zoom' in relayoutData else 6
    if input_id == 'reveal-best-location-button':
        threshold_value = df_aggregated['profitability'].quantile(0.97)
        return highlight_profitable_locations(rendered_fig, df_aggregated, threshold_value)
    if input_id == 'reset-button':
        rendered_fig = initial_fig
        return initial_fig
    elif input_id == 'recalculate-button':
        weight_factors = {
            'energy_production': wind_speed_weight,  # Adjust as needed
            'infrastructure_cost': infrastructure_cost_weight,  # Adjust as needed
            'environmental_factor': environmental_factor_weight,  # Adjust as needed
            'altitude_penalty': altitude_penalty  # Adjust as needed
        }
        # new_df = redefine_aggregation(df_wind, wind_speed_weight, infrastructure_cost_weight, environmental_factor_weight)
        new_df = redefine_aggregation(df_wind, weight_factors, dropdown_value)
        new_df['size_for_plot'] = new_df['profitability'].abs()
        updated_fig = px.scatter_mapbox(new_df, lat='Latitude', lon='Longitude',
                                    color='profitability', size='size_for_plot', 
                                    size_max=15, zoom=6, 
                                    center=dict(lat=map_center[0], lon=map_center[1]),
                                    mapbox_style="open-street-map",
                                    opacity=0.8)
        rendered_fig = updated_fig
        return updated_fig
    elif input_id == 'map' and clickData:
        updated_fig = plot_with_px(clicked_lon, clicked_lat, current_zoom)
        return updated_fig

    return initial_fig


if __name__ == '__main__':
    print('running the code')
    app.run_server(host='0.0.0.0', port=8080)
