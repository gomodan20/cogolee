from ultralytics import YOLO

def train_yolo_pose():
    # 모델 구성 yaml 파일 경로
    model_yaml = './models/custom.yaml'  # 상대경로로 바꾸세요
    
    # 데이터 구성 yaml 파일 경로
    data_yaml = './data/data.yaml'      # 상대경로로 바꾸세요
    
    # YOLO 객체 생성 (모델 구성 yaml 로드)
    model = YOLO(model_yaml)
    
    # 학습 실행
    model.train(
        data=data_yaml,
        epochs=100,
        patience=10,
        batch=16,
        imgsz=480,
        lr0=0.01,
        device='0',           # GPU 번호 (문자열 또는 int)
        pretrained=False,
        workers=4,
        amp=False,
        cache=True,
        save=True,
        project='./runs',
        name='exp5',
        save_period=1,
    )

if __name__ == '__main__':
    train_yolo_pose()
