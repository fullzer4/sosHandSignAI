from roboflow import Roboflow
rf = Roboflow(api_key="YG32Zd9Kcmc9muEmR5G4")
project = rf.workspace("senai-qb205").project("soshandsign")
dataset = project.version(1).download("yolov8")