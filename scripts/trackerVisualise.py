import json
import numpy as np

name = 'John Doe'
# my_details ={
#     'name': name,
#     'age' :29
#
# }

pcd = {
    "name": "Foo Bar",
    "age": 78,
    "friends": ["Jane", "John"],
    "balance": 345.80,
    "other_names": ("Doe", "Joe"),
    "active": True,
    "spouse": None
}
result_trks=[]
d = np.zeros([9,1])
obj_dict={"width":d[1], "height": d[0], "length": d[2], "x": d[3], "y": d[4], "z": d[5], "yaw": d[6], "id": d[7]}
result_trks.append(obj_dict)

total_list = ({"name": pcd ["name"], "objects": result_trks})
#tracker_json_outfile = "/media/yl/downloads/tracker_results/set_7/tracker_results_agexx_hitsxx_thresh_xx/tracker_px_stats_" + tracker_params +"_Q"+ q_params + ".json"
tracker_json_outfile = "person.json"

#if is_write:
with open(tracker_json_outfile, "w+") as outfile:
    json.dump(total_list, outfile, indent=1)
