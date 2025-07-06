# 🏌️ Golf Swing Labeling Automation using YOLOv8-Pose

YOLOv8-Pose 모델을 기반으로 골프 스윙 영상에서 자동으로 키포인트를 추출하고 바운딩 박스를 생성하는 라벨링 자동화 도구입니다.

---

## 📌 프로젝트 개요

골프 스윙 영상을 분석하여 프레임을 추출하고, 인체의 16개 관절(keypoints)을 감지하여 자동으로 라벨링을 수행합니다.  
수동 라벨링 과정을 최소화하고 시각화 이미지, 바운딩 박스, 키포인트 편집 기능까지 제공합니다.

---

## 📁 폴더 구조

```
yolo/labeling/
├── auto_labeling.py     # 메인 자동 라벨링 파이프라인
├── utils.py             # bbox 계산, 프레임 샘플링 등 유틸 함수
├── README.md            # 프로젝트 설명서
```

---

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/gomodan20/cogolee.git
cd cogolee

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

`requirements.txt`에 필요한 주요 라이브러리 예:
```
ultralytics
opencv-python
numpy
matplotlib
```

---

## 🎬 사용 방법

### 자동 라벨링 실행

```bash
python yolo/labeling/auto_labeling.py --video ./input/swing01.mp4 --output ./results/
```

- 입력: `.mp4` 스윙 영상
- 출력: 프레임 이미지, 키포인트 `.json`, 시각화 이미지

### ✍️ 키포인트 수동 수정

- 키포인트 수동 편집은 별도의 도구(keypoint_editor.py)를 포함하지 않습니다.
- AIHub 제공 키포인트 수정 도구를 사용 중입니다.

---

## 🧠 주요 기능

### 1. YOLOv8 Pose 기반 라벨링
- 16개 관절 키포인트 자동 추출
- 신뢰도(`confidence`) 기반 필터링

### 2. 프레임 샘플링
- 전체 프레임 중 80장 선택
  - 80장 미만 → 앞뒤 균등 반복
  - 80~120장 → 균등 샘플링
  - 120장 이상 → 핵심 중심 80장만 선택

### 3. 핵심 프레임 기반 자동 샘플링
- 손목 좌표 움직임을 기준으로:
  - 셋업:스윙:피니시 비율 = 20:40:20

### 4. 바운딩 박스 자동 생성
- 키포인트 기반 bbox 계산
- `.json`에 자동 삽입

### 5. 시각화 저장
- keypoint + bbox가 포함된 `.jpg` 저장
- 결과 폴더 구조 예:

```
legend_swing01/
├── image/      # 원본 프레임
├── json/       # 키포인트 라벨
├── visual/     # bbox & keypoint 시각화
```

---

## 🛠️ 커스터마이징

- 키포인트 이름 수정: `KEYPOINT_NAMES` 리스트 변경
- 프레임 수 조절: `auto_labeling.py` 내 `TARGET_FRAME_COUNT` 수정
- bbox margin 조절: `calculate_bbox_from_keypoints()` 함수의 `margin` 인자 변경
- 시각화 색상 변경: `visualize_frame()` 함수

---

## 📋 TODO

- [ ] YOLO + ST-GCN 연결 자동 파이프라인
- [ ] CLI 인자 더 정교하게 구성
- [ ] GUI 기반 편집 도구 개발
- [ ] 다중 인물 검출 대응

---

## 🤝 기여

Pull Request 및 Issue 생성을 환영합니다!

```bash
git checkout -b feature/yourFeature
git commit -m 'Add your feature'
git push origin feature/yourFeature
```

---

## 📄 라이선스

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.


---

## ⭐ 도움이 되셨다면 Star를 눌러주세요!
