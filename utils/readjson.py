import json
import os.path

f = open('../read.json', )

data = json.load(f)

for i in data['TruckTrack']:
    Cam = i["camName"]
    print(Cam)
    fromT = i["fromTime"]
    print(fromT)
    toT = i['toTime']
    print(toT)
    rx1, ry1, rx2, ry2 = i['ROI'].split(",")
    print(rx1, ry1, rx2, ry2)
    path = os.path.join('C:/Users/munda/PycharmProjects/IAmSmart-T0/images/', Cam, fromT.split("T")[0])

    image_path = os.listdir(path)
    for each_files in image_path:
        if fromT < each_files < toT:
            print("Inside", each_files)
        else:
            print("Re", each_files)

import json


# function to add to JSON


y = {"emp_name": "Nikhil",
     "email": "nikhil@geeksforgeeks.org",
     "job_profile": "Full Time"
     }

write_json(y)
f.close()
