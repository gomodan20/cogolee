``` 
YOLO v8 Pose Validation
```

from ultralytics import YOLO

# 1. 모델 로드
model = YOLO(r"D:\resized_dataset2\resized_dataset5\best.pt")

# 2. 예측 실행
model.predict(
    source=r"D:\resized_dataset2\resized_dataset5\predict\images",  # 예측할 이미지 폴더
    save=True,                   # 결과 이미지 저장
    save_txt=True,               # 결과 라벨(txt) 저장
    project=r"D:\resized_dataset2\resized_dataset5",  # 결과 저장할 상위 폴더
    name="predict_result",       # 결과가 저장될 하위 폴더명
    exist_ok=True,                # 이미 폴더가 있어도 덮어쓰기 허용
    batch=16,              # batch size 조정
    conf=0.5               # confidence threshold 조정
)