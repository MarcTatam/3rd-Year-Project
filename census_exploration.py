import keplergl
import json

def open_census_isat():
    with open("milano_istat.json","r") as f:
        struct = json.load(f)
    this_map = keplergl.KeplerGl()
    this_map.add_data(struct)
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
    this_map.save_to_html(data=struct, 
	file_name='Census 1.html',config=this_map.config)


if __name__ == "__main__":
    open_census_isat()