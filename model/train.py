from ultralytics import YOLO
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
model = YOLO('yolov8n.yaml')

model.train(data='./soshandsign-1/data.yaml', epochs=100, imgsz=[1280,720]) 