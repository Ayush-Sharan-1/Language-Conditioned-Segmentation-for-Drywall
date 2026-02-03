import os
from roboflow import Roboflow

rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("Ayush-Lab").project("drywall-join-detect-qq7mk")
dataset = project.version(1).download("coco")

