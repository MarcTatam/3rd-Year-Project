import plotly.graph_objects as go
import plotly.express as ex
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open_cdr as cdr
import geojson

def graph_day_one():
    df = cdr.merge_countries("2013-11-01")
    selected_df = df.loc[df[0] == 1]
    selected_df[1] = pd.to_datetime(selected_df[1],unit='ms')
    print(selected_df)
    selected_df.plot(x=1, y=[3,4,5,6,7])
    plt.show()

def heatmap():
    df = cdr.merge_countries("2013-12-11")
    df = df[df[1] == 1386788400000]
    df.columns =['CellID', 'Timestamp', 'SMSIN','SMSOUT','CALLIN','CALLOUT','INTERNET']
    df['CellID'] = df['CellID'] -1
    df['SMSIN'] = np.log2(df['SMSIN'])
    print(df[df['CellID']==5161])
    with open("milano-grid.geojson") as f:
        gj = geojson.load(f)
    fig= go.Figure(go.Choroplethmapbox(
                            geojson=gj,
                            locations=df.CellID,
                            z=df.SMSIN,
                            colorscale="jet",  
                            marker_line_width=0,

                            marker=dict(opacity=0.5)
                           ))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=9.19, mapbox_center_lat=45.4642,mapbox_zoom=10)
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(colorbar_title_text="Test")
    fig.show()
    #fig.write_image("./images/fig1.png")

def develop_image(timestamp:int):
    dt = datetime.fromtimestamp(int(timestamp/1000)) + timedelta(hours = 1)
    df = cdr.merge_countries(dt.strftime('%Y-%m-%d'))
    df = df[df[1] == timestamp]
    df.columns =['CellID', 'Timestamp', 'SMSIN','SMSOUT','CALLIN','CALLOUT','INTERNET']
    df['CellID'] = df['CellID'] -1
    df['SMSIN'] = np.log10(df['SMSIN'])
    with open("milano-grid.geojson") as f:
        gj = geojson.load(f)
    fig= go.Figure(go.Choroplethmapbox(
                            geojson=gj,
                            locations=df.CellID,
                            z=df.SMSIN,
                            zmin = -5,
                            zmax = 3.4,
                            colorscale="jet",  
                            marker_line_width=0,
                            marker=dict(opacity=0.5)
                           ))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=9.19, mapbox_center_lat=45.4642,mapbox_zoom=10)
    fig.update_geos(fitbounds="locations")
    fig.update_layout(title_text="Heatmap of SMS Activity " + dt.strftime('%Y-%m-%d %H:%M'))
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.update_traces(colorbar_title_text="Log(Activity)")
    fig.add_annotation(text = "SMS Activity: "+dt.strftime('%Y-%m-%d %H:%M'), x=0.2, y=-0.021, bgcolor="white")
    fig.write_image("./images/"+dt.strftime('sms%Y-%m-%d-%H-%M')+".png")

def develop_images(start:int, stop:int):
    this = start
    while this <= stop:
        print(this)
        develop_image(this)
        this += 600000
if __name__ == "__main__":
    develop_images(1387209600000, 1388616600000)

