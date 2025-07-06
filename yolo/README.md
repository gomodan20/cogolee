# 1. 📌 Project Title
- 골프 자세 분석을 위한 YOLO v8 Pose 모델 학습

## 2. 📝 Overview / Summary
- 프로젝트 간략 설명
- YOLOv8 Pose로 사람의 키포인트 추출 → ST-GCN 포즈추정에 사용

## 3. 📁 Directory Structure
yolo/

├── 0527_yolo_train2.ipynb              
├── evaluate_yolo_model.ipynb      
├── yolo_validation.py 
├── custom.yaml            # 학습 설정
├── data.yaml
└── results/
    ├── YOLO_Pose_mAP50-95.png
    ├── Training_and_Validation_Loss.png
    └── golf_pose_validation_test_comparison.png

## 4. ⚙️ Dependencies
pip install ultralytics
pip install opencv-python
pip install numpy

### 5. 🏋️ Training Instructions
For Colab training: [YOLOv8_Pose_Training_Notebook](./yolo/0527_yolo_train2.ipynb)

#### 6. 🧪 Evaluation / Testing
python evaluate.py --weights best.pt --data config.yaml --imgsz 640

##### 7. 🎯 Results
- mAP: mAP50-95(B)- 0.971 | mAP50-95(P)- 0.88
