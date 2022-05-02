import keplergl
import pandas as pd
import geojson as gj
import json
import pickle

def map_events():
    """Maps the events"""
    this_map = keplergl.KeplerGl()
    venues = pd.read_csv("Venues.csv")
    events = pd.read_csv("Events.csv")
    df = pd.merge(venues, events, on=["Venue"])
    df[['Long','Lat']] = df.Coords.str.split(", ",expand=True,)
    count = {}
    events_json = {
        "type": "FeatureCollection",
        "features": []
        }
    for index, Row in df.iterrows():
        if Row["Venue"] in count.keys():
            elevation = 50*count[Row["Venue"]]
            count[Row["Venue"]] += 1
        else:
            elevation = 0
            count[Row["Venue"]] = 1
        point = {}
        point["type"] = "Feature"
        point["geometry"] = {
            "type": "Point",
            "coordinates": [float(Row["Lat"]), float(Row["Long"]),elevation]}
        point["properties"] = {"Venue": Row["Venue"], "Event" : Row["Name"], "Date" : Row["Date"]}
        events_json["features"].append(point)
    this_map.add_data(events_json, "Events")
    this_map.config = {'version': 'v1',
 'config': {'visState': {'filters': [],
   'layers': [{'id': '6695y74',
     'type': 'geojson',
     'config': {'dataId': 'Events',
      'label': 'Events',
      'color': [76, 154, 78],
      'highlightColor': [252, 242, 26, 255],
      'columns': {'geojson': '_geojson'},
      'isVisible': True,
      'visConfig': {'opacity': 0.8,
       'strokeOpacity': 0.8,
       'thickness': 0.5,
       'strokeColor': None,
       'colorRange': {'name': 'Global Warming',
        'type': 'sequential',
        'category': 'Uber',
        'colors': ['#5A1846',
         '#900C3F',
         '#C70039',
         '#E3611C',
         '#F1920E',
         '#FFC300']},
       'strokeColorRange': {'name': 'Global Warming',
        'type': 'sequential',
        'category': 'Uber',
        'colors': ['#5A1846',
         '#900C3F',
         '#C70039',
         '#E3611C',
         '#F1920E',
         '#FFC300']},
       'radius': 10,
       'sizeRange': [0, 10],
       'radiusRange': [0, 50],
       'heightRange': [0, 500],
       'elevationScale': 5,
       'enableElevationZoomFactor': True,
       'stroked': False,
       'filled': True,
       'enable3d': False,
       'wireframe': False},
      'hidden': False,
      'textLabel': [{'field': None,
        'color': [255, 255, 255],
        'size': 18,
        'offset': [0, 0],
        'anchor': 'start',
        'alignment': 'center'}]},
     'visualChannels': {'colorField': None,
      'colorScale': 'quantile',
      'strokeColorField': None,
      'strokeColorScale': 'quantile',
      'sizeField': None,
      'sizeScale': 'linear',
      'heightField': None,
      'heightScale': 'linear',
      'radiusField': None,
      'radiusScale': 'linear'}}],
   'interactionConfig': {'tooltip': {'fieldsToShow': {'Events': [{'name': 'Venue',
        'format': None},
       {'name': 'Event', 'format': None},
       {'name': 'Date', 'format': None}]},
     'compareMode': False,
     'compareType': 'absolute',
     'enabled': True},
    'brush': {'size': 0.5, 'enabled': False},
    'geocoder': {'enabled': False},
    'coordinate': {'enabled': False}},
   'layerBlending': 'normal',
   'splitMaps': [],
   'animationConfig': {'currentTime': None, 'speed': 1}},
  'mapState': {'bearing': 0,
   'dragRotate': False,
   'latitude': 45.455934682435,
   'longitude': 9.23461965432608,
   'pitch': 0,
   'zoom': 10.451941083083048,
   'isSplit': False},
  'mapStyle': {'styleType': 'dark',
   'topLayerGroups': {},
   'visibleLayerGroups': {'label': True,
    'road': True,
    'border': False,
    'building': True,
    'water': True,
    'land': True,
    '3d building': False},
   'threeDBuildingColor': [9.665468314072013,
    17.18305478057247,
    31.1442867897876],
   'mapStyles': {}}}}
    this_map.save_to_html(file_name="Events.html")

def map_venues():
    """Maps the venues"""
    this_map = keplergl.KeplerGl()
    venues = pd.read_csv("Venues.csv")
    venues[['Long','Lat']] = venues.Coords.str.split(", ",expand=True,)
    venues_json= {
        "type": "FeatureCollection",
        "features": []
        }
    i = 0
    for index, Row in venues.iterrows():
        i += 1
        point = {}
        point["type"] = "Feature"
        point["geometry"] = {
            "type": "Point",
            "coordinates": [float(Row["Lat"]), float(Row["Long"])]}
        point["properties"] = {"name": Row["Name"]}
        venues_json["features"].append(point)
    print(venues_json)
    this_map.add_data(venues_json, "Venues")
    this_map.save_to_html(file_name="Venues.html")

def map_nn_events():
    events_json = {
        "type": "FeatureCollection",
        "features": []
        }
    with open("cellevents.pkl", "rb") as f:
        cell_events = pickle.load(f)
    cells = {}
    with open("milano-grid.geojson", "r") as f:
        for cell in json.load(f)["features"]:
            cells[cell["properties"]["cellId"]] = [(cell["geometry"]["coordinates"][0][0][0]+cell["geometry"]["coordinates"][0][2][0])/2,(cell["geometry"]["coordinates"][0][0][1]+cell["geometry"]["coordinates"][0][2][1])/2]
    for cell in cell_events.keys():
        for i in range(len(cell_events[cell])):
            point = {}
            point["type"] = "Feature"
            point["geometry"] = {
                "type": "Point",
                "coordinates": cells[cell]+[50*i]}
            point["properties"] = {"CellID": cell,
                                   "Day" : cell_events[cell][i]}
            events_json["features"].append(point)
    this_map = keplergl.KeplerGl()
    this_map.add_data(events_json, "NN Detected Events")
    this_map.config = {
	'version': 'v1',
	'config': {
		'mapState': {
			'bearing': 0,
			'dragRotate': False,
			'latitude': 45.455934682435,
			'longitude': 9.23461965432608,
			'pitch': 0,
			'zoom': 10.451941083083048,
			'isSplit': False
		},
		'mapStyle': {
			'styleType': 'dark',
			'topLayerGroups': {},
			'visibleLayerGroups': {
				'label': True,
				'road': True,
				'border': False,
				'building': True,
				'water': True,
				'land': True,
				'3d building': False
			},
			'threeDBuildingColor': [9.665468314072013,
				17.18305478057247,
				31.1442867897876
			],
			'mapStyles': {}
		}
	}
}
    this_map.save_to_html(file_name="NNevents.html")

def map_nn_events_alternate():
    events_json = {
        "type": "FeatureCollection",
        "features": []
        }
    with open("celleventssingle.pkl", "rb") as f:
        cell_events = pickle.load(f)
    cells = {}
    with open("milano-grid.geojson", "r") as f:
        for cell in json.load(f)["features"]:
            cells[cell["properties"]["cellId"]] = [(cell["geometry"]["coordinates"][0][0][0]+cell["geometry"]["coordinates"][0][2][0])/2,(cell["geometry"]["coordinates"][0][0][1]+cell["geometry"]["coordinates"][0][2][1])/2]
    for cell in cell_events.keys():
        for i in range(len(cell_events[cell])):
            point = {}
            point["type"] = "Feature"
            point["geometry"] = {
                "type": "Point",
                "coordinates": cells[cell]+[50*i]}
            point["properties"] = {"CellID": cell,
                                   "Day" : cell_events[cell][i]}
            events_json["features"].append(point)
    this_map = keplergl.KeplerGl()
    this_map.add_data(events_json, "NN Detected Events")
    this_map.config = {
	'version': 'v1',
	'config': {
		'mapState': {
			'bearing': 0,
			'dragRotate': False,
			'latitude': 45.455934682435,
			'longitude': 9.23461965432608,
			'pitch': 0,
			'zoom': 10.451941083083048,
			'isSplit': False
		},
		'mapStyle': {
			'styleType': 'dark',
			'topLayerGroups': {},
			'visibleLayerGroups': {
				'label': True,
				'road': True,
				'border': False,
				'building': True,
				'water': True,
				'land': True,
				'3d building': False
			},
			'threeDBuildingColor': [9.665468314072013,
				17.18305478057247,
				31.1442867897876
			],
			'mapStyles': {}
		}
	}
}
    this_map.save_to_html(file_name="NNeventsAlternate.html")

def map_nn_events_combined():
    events_json = {
        "type": "FeatureCollection",
        "features": []
        }
    with open("combinedevents.pkl", "rb") as f:
        cell_events = pickle.load(f)
    cells = {}
    with open("milano-grid.geojson", "r") as f:
        for cell in json.load(f)["features"]:
            cells[cell["properties"]["cellId"]] = [(cell["geometry"]["coordinates"][0][0][0]+cell["geometry"]["coordinates"][0][2][0])/2,(cell["geometry"]["coordinates"][0][0][1]+cell["geometry"]["coordinates"][0][2][1])/2]
    for cell in cell_events.keys():
        for i in range(len(cell_events[cell])):
            point = {}
            point["type"] = "Feature"
            point["geometry"] = {
                "type": "Point",
                "coordinates": cells[cell]+[50*i]}
            point["properties"] = {"CellID": cell,
                                   "Day" : cell_events[cell][i]}
            events_json["features"].append(point)
    this_map = keplergl.KeplerGl()
    this_map.add_data(events_json, "NN Detected Events")
    this_map.config = {
	'version': 'v1',
	'config': {
		'mapState': {
			'bearing': 0,
			'dragRotate': False,
			'latitude': 45.455934682435,
			'longitude': 9.23461965432608,
			'pitch': 0,
			'zoom': 10.451941083083048,
			'isSplit': False
		},
		'mapStyle': {
			'styleType': 'dark',
			'topLayerGroups': {},
			'visibleLayerGroups': {
				'label': True,
				'road': True,
				'border': False,
				'building': True,
				'water': True,
				'land': True,
				'3d building': False
			},
			'threeDBuildingColor': [9.665468314072013,
				17.18305478057247,
				31.1442867897876
			],
			'mapStyles': {}
		}
	}
}
    this_map.save_to_html(file_name="NNeventsCombined.html")

def map_nn_events_double():
    events_json = {
        "type": "FeatureCollection",
        "features": []
        }
    with open("doubleevents.pkl", "rb") as f:
        cell_events = pickle.load(f)
    cells = {}
    with open("milano-grid.geojson", "r") as f:
        for cell in json.load(f)["features"]:
            cells[cell["properties"]["cellId"]] = [(cell["geometry"]["coordinates"][0][0][0]+cell["geometry"]["coordinates"][0][2][0])/2,(cell["geometry"]["coordinates"][0][0][1]+cell["geometry"]["coordinates"][0][2][1])/2]
    for cell in cell_events.keys():
        for i in range(len(cell_events[cell])):
            point = {}
            point["type"] = "Feature"
            point["geometry"] = {
                "type": "Point",
                "coordinates": cells[cell]+[50*i]}
            point["properties"] = {"CellID": cell,
                                   "Day" : cell_events[cell][i]}
            events_json["features"].append(point)
    this_map = keplergl.KeplerGl()
    this_map.add_data(events_json, "NN Detected Events")
    this_map.config = {
	'version': 'v1',
	'config': {
		'mapState': {
			'bearing': 0,
			'dragRotate': False,
			'latitude': 45.455934682435,
			'longitude': 9.23461965432608,
			'pitch': 0,
			'zoom': 13.451941083083048,
			'isSplit': False
		},
		'mapStyle': {
			'styleType': 'dark',
			'topLayerGroups': {},
			'visibleLayerGroups': {
				'label': True,
				'road': True,
				'border': False,
				'building': True,
				'water': True,
				'land': True,
				'3d building': False
			},
			'threeDBuildingColor': [9.665468314072013,
				17.18305478057247,
				31.1442867897876
			],
			'mapStyles': {}
		}
	}
}
    this_map.save_to_html(file_name="NNeventsDouble.html")


if __name__ == "__main__":
    map_events()
