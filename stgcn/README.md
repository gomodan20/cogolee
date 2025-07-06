# Golf Swing Classification using ST-GCN

ST-GCN (Spatial-Temporal Graph Convolutional Networks)을 사용하여 골프 스윙 동작을 분류하는 딥러닝 모델입니다.

## 🏌️ 프로젝트 개요

골프 스윙의 포즈 시퀀스 데이터를 분석하여 다양한 스윙 유형을 자동으로 분류합니다. 인체의 16개 키포인트를 사용하여 시공간적 특징을 학습합니다.

## 📁 프로젝트 구조

```
stgcn/
├── config.py              # 설정 파일
├── dataset.py             # 데이터셋 클래스 및 데이터 로딩
├── loss_functions.py      # 커스텀 손실 함수
├── training_utils.py      # 훈련 및 평가 유틸리티
├── visualization.py       # 시각화 도구
├── train.py               # 메인 훈련 스크립트
├── STGCN.py               # ST-GCN 모델 스켈레톤
├── requirements.txt       # 필요한 패키지 목록
└── README.md             # 프로젝트 설명서
```

## 🚀 설치 및 설정

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/your-username/golf-swing-classification.git
cd golf-swing-classification

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 데이터 준비

데이터는 다음 형식으로 준비되어야 합니다:
- `{swing_id}_seq.pt`: 포즈 시퀀스 데이터 [C, T, V] 형태
- `{swing_id}_label.pt`: 해당 스윙의 라벨 (정수)

여기서:
- C: 채널 수 (x, y 좌표 등)
- T: 시간 프레임 수
- V: 키포인트 수 (16개)

### 3. 설정 수정

`config.py` 파일에서 데이터 경로와 훈련 파라미터를 수정하세요:

```python
DEFAULT_PATHS = {
    'data_root_dir': "your/data/path",
    'model_save_dir': "your/model/save/path"
}
```

## 🏃‍♂️ 사용 방법

### 기본 훈련 실행

```bash
python train.py
```

### 커스텀 설정으로 훈련

```python
from train import train_model

results = train_model(
    data_root_dir="path/to/your/data",
    model_save_dir="path/to/save/models",
    project_name="my_experiment",
    weighted=True,          # 클래스 불균형 처리
    batch_size=32,
    num_epochs=200,
    num_class=3
)
```

## 📊 주요 기능

### 1. 클래스 불균형 처리
- **Weighted Sampling**: 클래스별 샘플 수에 따른 가중치 적용
- **Focal Loss**: 어려운 샘플에 더 집중하는 손실 함수

### 2. 자동 데이터 분할
- 훈련/검증/테스트 데이터 자동 분할
- 재현 가능한 랜덤 시드 설정

### 3. 훈련 모니터링
- 실시간 손실 및 정확도 추적
- Early Stopping으로 과적합 방지
- 학습률 스케줄링

### 4. 결과 시각화
- Confusion Matrix 생성
- 훈련 히스토리 플롯
- 분류 성능 리포트

## 🏌️ 키포인트 정의

모델은 다음 16개 인체 키포인트를 사용합니다:

```
0: head           8: left_wrist
1: neck           9: hip (center)
2: chest         10: right_hip
3: right_shoulder 11: left_hip
4: left_shoulder  12: right_knee
5: right_elbow    13: left_knee
6: left_elbow     14: right_ankle
7: right_wrist    15: left_ankle
```

## 📈 모델 성능

모델 훈련 후 다음 결과물이 생성됩니다:
- `best_val_acc_model.pt`: 최고 검증 정확도 모델
- `best_val_loss_model.pt`: 최저 검증 손실 모델
- `train_log.csv`: 훈련 로그
- `confusion_matrix_*.png`: 혼동 행렬 시각화
- `training_history_*.png`: 훈련 히스토리 플롯

## ⚙️ 설정 옵션

### 모델 설정
```python
MODEL_CONFIG = {
    'in_channels': 2,
    'num_class': 3,
    'edge_importance_weighting': True,
    'dropout': 0.3
}
```

### 훈련 설정
```python
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 200,
    'learning_rate': 0.0005,
    'early_stop_patience': 20
}
```

## 🛠️ 커스터마이징

### 새로운 손실 함수 추가
`loss_functions.py`에 새로운 손실 함수를 구현하고 `train.py`에서 사용할 수 있습니다.

### 데이터 전처리 수정
`dataset.py`의 `GolfSwingDataset` 클래스를 수정하여 데이터 전처리 로직을 변경할 수 있습니다.

### 시각화 커스터마이징
`visualization.py`에서 플롯 스타일이나 추가 시각화를 구현할 수 있습니다.

## 📋 TODO

- [ ] 모델 앙상블 구현
- [ ] 실시간 추론 스크립트 추가
- [ ] 웹 인터페이스 개발
- [ ] 더 많은 평가 메트릭 추가
- [ ] 하이퍼파라미터 자동 튜닝

## 🤝 기여

프로젝트 개선을 위한 기여를 환영합니다!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 문의

문제가 있거나 질문이 있으시면 이슈를 생성해 주세요.

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!
