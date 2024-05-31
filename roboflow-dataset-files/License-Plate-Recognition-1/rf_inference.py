#!/usr/bin/env python3

from roboflow import Roboflow

rf = Roboflow(api_key="")

project = rf.workspace("lpr-ydttq").project("license-plate-recognition-bgpr1")

model = project.version(1, local="http://localhost:9001/").model

prediction = model.predict("/home/kelvin/projects/LPR/test_plate.jpg")

print(prediction.json())

prediction.save("result.jpg")

prediction.plot()