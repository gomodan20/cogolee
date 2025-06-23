#!/bin/bash

# 데이터 압축 해제 (로컬에 이미 있으면 스킵 가능)
unzip -o ./data/resized_dataset5.zip -d ./data/

# ultralytics 설치 (가상환경에서 한 번만 하면 됨)
pip install ultralytics

# 학습 실행
yolo pose train data=./data/data.yaml model=./models/custom.yaml epochs=100 patience=10 batch=16 imgsz=480 lr0=0.01 device=0 pretrained=False workers=4 amp=False cache=True save=True project=./runs name=exp5 save_period=1
