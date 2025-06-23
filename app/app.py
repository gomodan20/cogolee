from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
import json
from datetime import datetime
import traceback
import gc
import base64
import io
from flask_migrate import Migrate
import draw

# Flask 앱 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///golf_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# SQLAlchemy 및 Migrate 객체 생성
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# 업로드 폴더 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# 로그인 매니저 설정
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 키포인트 매핑
KEYPOINT_NAMES = [
    "head", "neck", "chest",
    "right_shoulder", "left_shoulder",
    "right_elbow", "left_elbow",
    "right_wrist", "left_wrist",
    "hip", "right_hip", "left_hip",
    "right_knee", "left_knee",
    "right_ankle", "left_ankle"
]

# 클래스 매핑
CLASS_NAMES = ['amateur', 'semipro', 'pro']

# 데이터베이스 모델
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.now)
    videos = db.relationship('VideoAnalysis', backref='user', lazy=True)

class VideoAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    stored_filename = db.Column(db.String(255))     # 사용자가 지정한 영상 이름
    prediction_result = db.Column(db.String(50))
    keypoints_data = db.Column(db.Text)  # JSON 형태로 저장
    upload_time = db.Column(db.DateTime, default=datetime.now)
    analysis_time = db.Column(db.DateTime)
    frame_count = db.Column(db.Integer)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# 모델 클래스들 (원본 코드에서 가져온 함수들)
# 모델 클래스들 (원본 코드에서 가져온 함수들)
# 모델 클래스들 (원본 코드에서 가져온 함수들)
class GolfAnalysisModel:
    def __init__(self):
        self.yolo_model = None
        self.stgcn_model = None
        self.device = None
        
    def initialize_models(self):
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"사용 디바이스: {self.device}")
            
            # YOLO 모델 로드
            yolo_path = os.path.join(app.root_path, 'model', 'best_resized_yolo_최종.pt')
            self.yolo_model = YOLO(yolo_path)
            print("✅ YOLO 모델 로드 완료")
            
            # STGCN 모델 로드
            try:
                import STGCN
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                self.stgcn_model = STGCN.Model(
                    in_channels=2,
                    num_class=3,
                    graph_args={'layout': 'golf', 'strategy': 'spatial'}
                )
                
                stgcn_path = os.path.join(app.root_path, 'model', 'best_stgcn_3class.pt')
                if os.path.exists(stgcn_path):
                    checkpoint = torch.load(stgcn_path, map_location='cpu')
                    self.stgcn_model.load_state_dict(checkpoint)
                    self.stgcn_model = self.stgcn_model.to(self.device)
                    self.stgcn_model.eval()
                    print("✅ STGCN 모델 로드 완료")
                else:
                    print(f"❌ 모델 파일을 찾을 수 없습니다: {stgcn_path}")
                    self.stgcn_model = None
            except Exception as e:
                print(f"STGCN 모델 로드 실패: {e}")
                self.stgcn_model = None
                
        except Exception as e:
            print(f"모델 초기화 중 오류 발생: {e}")
            raise e

# 전역 모델 인스턴스
golf_model = GolfAnalysisModel()

# 유틸리티 함수들 (원본 코드에서 가져옴)
def clamp(val, minimum, maximum):
    return max(minimum, min(val, maximum))

def score_to_visibility(score, threshold=0.3):
    return 2 if score > threshold else 1 if score > 0.05 else 0

def calculate_bbox_from_keypoints(keypoints, margin=10):
    visible_points = []
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v > 0:
            visible_points.append([x, y])
    if len(visible_points) < 5:
        return None
    visible_points = np.array(visible_points)
    x_min = int(np.min(visible_points[:, 0]))
    y_min = int(np.min(visible_points[:, 1]))
    x_max = int(np.max(visible_points[:, 0]))
    y_max = int(np.max(visible_points[:, 1]))
    return [x_min - margin, y_min - margin, x_max + margin, y_max + margin]

def center_crop_and_resize(img, bbox, target_h_ratio=0.5, output_size=(640, 360)):
    ih, iw = img.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    box_h = y2 - y1
    scale = (output_size[1] * target_h_ratio) / box_h
    new_w, new_h = int(iw * scale), int(ih * scale)
    resized = cv2.resize(img, (new_w, new_h))
    rcx, rcy = int(cx * scale), int(cy * scale)
    left = rcx - output_size[0] // 2
    top = rcy - output_size[1] // 2
    right = left + output_size[0]
    bottom = top + output_size[1]
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - resized.shape[1])
    pad_bottom = max(0, bottom - resized.shape[0])
    padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
    cropped = padded[top+pad_top:bottom+pad_top, left+pad_left:right+pad_left]
    return cropped

def calculate_left_wrist_mv(keypoints_data):
    movements = []
    prev_left_wrist_x = None
    prev_left_wrist_y = None
    
    for frame_number, frame_dict in enumerate(keypoints_data):
        kp = frame_dict['keypoints']
        
        if len(kp) > 8:
            now_left_wrist_x, now_left_wrist_y = kp[8]
            
            if prev_left_wrist_x is not None and prev_left_wrist_y is not None:
                movement = math.sqrt((now_left_wrist_x - prev_left_wrist_x) ** 2 +
                                     (now_left_wrist_y - prev_left_wrist_y) ** 2)
                movements.append(movement)
            else:
                movements.append(0)
            
            prev_left_wrist_x = now_left_wrist_x
            prev_left_wrist_y = now_left_wrist_y
        else:
            movements.append(0)
    
    return movements

def uniform_sample(sequence, target_length):
    try:
        if len(sequence) == target_length:
            return sequence
        if len(sequence) == 0:
            return []
        idxs = np.linspace(0, len(sequence) - 1, target_length).astype(int)
        return [sequence[i] for i in idxs]
    except Exception as e:
        print(f"균일 샘플링 오류: {e}")
        return sequence

def normalize_yolo_pose_keypoints(keypoints, box=None, h=None, center_x=None, center_y=None):
    """
    YOLO 키포인트를 정규화하는 함수 - 두 번째 코드의 로직 적용
    """
    if isinstance(keypoints, list):
        keypoints = np.array(keypoints)

    if len(keypoints) == 0:
        if h is None:
            return None
        else:
            return torch.zeros((2, 16), dtype=torch.float32)
    
    keypoints = np.array(keypoints)

    # 키포인트가 이미 (x, y) 형태인지 (x, y, v) 형태인지 확인
    if keypoints.shape[1] == 3:
        xys = keypoints[:, :2]  # (x, y)만 추출
    elif keypoints.shape[1] == 2:
        xys = keypoints
    else:
        raise ValueError(f"예상치 못한 키포인트 shape: {keypoints.shape}")

    # 정규화 기준값이 없으면 계산해서 반환
    if h is None or center_x is None or center_y is None:
        # 유효한 키포인트만 필터링 (0보다 큰 좌표)
        valid_mask = (xys[:, 0] > 0) & (xys[:, 1] > 0)
        valid_points = xys[valid_mask]
        
        if len(valid_points) < 3:
            print("유효한 키포인트가 부족합니다.")
            return None
            
        # 사람 높이 계산 (더 robust하게)
        min_y = np.min(valid_points[:, 1])
        max_y = np.max(valid_points[:, 1])
        person_height = max_y - min_y
        
        if person_height < 50:  # 최소 높이 50픽셀
            print(f"person_height가 너무 작습니다: {person_height}")
            return None
        
        # 중심 계산
        center_x = np.mean(valid_points[:, 0])
        center_y = np.mean(valid_points[:, 1])
        
        # 정규화 정보만 반환 (두 번째 코드 방식)
        return person_height, center_x, center_y

    # === 실제 정규화 수행 ===
    # print(f"\n=== 정규화 디버깅 ===")
    # print(f"h (height): {h}, type: {type(h)}")
    # print(f"center_x: {center_x}, type: {type(center_x)}")
    # print(f"center_y: {center_y}, type: {type(center_y)}")
    # print(f"첫 번째 키포인트 원본: {xys[0]}")
    
    # 정규화 수행
    normalized_xys = xys.copy().astype(np.float32)  # dtype 명시적 지정
    
    # 단계별 계산 확인
    # print(f"\n=== 단계별 계산 ===")
    x_orig, y_orig = xys[0, 0], xys[0, 1]
    # print(f"원본 좌표: x={x_orig}, y={y_orig}")
    # print(f"x - center_x: {x_orig} - {center_x} = {x_orig - center_x}")
    # print(f"y - center_y: {y_orig} - {center_y} = {y_orig - center_y}")
    # print(f"(x - center_x) / h: ({x_orig - center_x}) / {h} = {(x_orig - center_x) / h}")
    # print(f"(y - center_y) / h: ({y_orig - center_y}) / {h} = {(y_orig - center_y) / h}")
    
    # 정규화 수행
    normalized_xys[:, 0] = (xys[:, 0] - center_x) / h
    normalized_xys[:, 1] = (xys[:, 1] - center_y) / h
    
    # 디버깅: 정규화된 값 확인
    print(f"\n=== 정규화 결과 ===")
    print(f"정규화 전 샘플: {xys[0]}")
    print(f"정규화 후 샘플: {normalized_xys[0]}")
    print(f"normalized_xys dtype: {normalized_xys.dtype}")
    print(f"정규화 범위: x=[{np.min(normalized_xys[:, 0]):.3f}, {np.max(normalized_xys[:, 0]):.3f}], y=[{np.min(normalized_xys[:, 1]):.3f}, {np.max(normalized_xys[:, 1]):.3f}]")
    
    # 0이 되는지 체크
    zero_count = np.sum(normalized_xys == 0)
    print(f"0인 값의 개수: {zero_count}/{normalized_xys.size}")
    
    # 트랜스포즈하여 (2, 16) 형태로 변환
    keypoints_xy = torch.tensor(normalized_xys.T, dtype=torch.float32)  # (2, 16)
    
    print(f"최종 tensor shape: {keypoints_xy.shape}")
    print(f"최종 tensor 샘플: {keypoints_xy[:, 0]}")
    
    return keypoints_xy

# analyze_golf_swing 함수에서 사용할 수 있는 보조 함수
def extract_yolo_keypoints_safely(result):
    """
    YOLO 결과에서 키포인트를 안전하게 추출
    """
    frame_keypoints = []
    
    if result and hasattr(result[0], 'keypoints') and result[0].keypoints is not None:
        if len(result[0].keypoints.xy) > 0:
            pts = result[0].keypoints.xy[0].cpu().numpy()  # (17, 2) 형태
            confs = result[0].keypoints.conf[0].cpu().numpy() if hasattr(result[0].keypoints, 'conf') else None
            
            # COCO 17개 키포인트 중 16개만 사용 (골프 모델에 맞게)
            for i, (x, y) in enumerate(pts[:16]):  # 16개만
                # confidence가 있으면 확인, 없으면 좌표가 0이 아닌지만 확인
                is_valid = True
                if confs is not None:
                    is_valid = confs[i] > 0.3  # confidence threshold
                else:
                    is_valid = x > 0 and y > 0  # 좌표가 유효한지만 확인
                
                if is_valid:
                    frame_keypoints.append((float(x), float(y)))
                else:
                    frame_keypoints.append((0.0, 0.0))  # 무효한 키포인트는 0으로
    
    # 16개 키포인트가 안 되면 0으로 패딩
    while len(frame_keypoints) < 16:
        frame_keypoints.append((0.0, 0.0))
    
    return frame_keypoints[:16]  # 정확히 16개만 반환

def convert_to_stgcn_tensor(core_keypoints):

    """
    키포인트 데이터를 STGCN 텐서로 변환 - YOLO 데이터 구조에 맞게 수정
    """
    try:
        if not core_keypoints:
            print("경고: 빈 키포인트 데이터입니다.")
            return None
            
        num_frames = len(core_keypoints)
        num_joints = 16
        
        data = np.zeros((num_frames, num_joints, 2), dtype=np.float32)
        
        h, center_x, center_y = None, None, None
        
        print(f"키포인트 데이터 구조 확인:")
        print(f"총 프레임 수: {num_frames}")
        if core_keypoints:
            print(f"첫 번째 프레임 구조: {core_keypoints[0].keys()}")
            if 'keypoints' in core_keypoints[0]:
                print(f"키포인트 수: {len(core_keypoints[0]['keypoints'])}")
                print(f"키포인트 샘플: {core_keypoints[0]['keypoints'][:3]}")
        
        # 정규화 기준 추출 - YOLO 데이터 구조에 맞게 수정
        for i, frame_data in enumerate(core_keypoints):
            if 'keypoints' in frame_data and frame_data['keypoints']:
                keypoints_list = frame_data['keypoints']
                
                # YOLO에서 온 키포인트는 [(x1, y1), (x2, y2), ...] 형태
                if len(keypoints_list) >= 16:
                    # tuple 리스트를 numpy 배열로 변환
                    keypoints = np.array(keypoints_list[:16], dtype=np.float32)
                    
                    print(f"프레임 {i} 키포인트 shape: {keypoints.shape}")
                    print(f"프레임 {i} 키포인트 샘플: {keypoints[:3]}")
                    
                    result = normalize_yolo_pose_keypoints(keypoints)
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        h, center_x, center_y = result
                        print(f"정규화 정보를 프레임 {i}에서 추출했습니다.")
                        print(f"h={h:.2f}, center_x={center_x:.2f}, center_y={center_y:.2f}")
                        break
        
        if h is None or center_x is None or center_y is None:
            print("정규화 기준 추출 실패, 기본값 사용")
            h, center_x, center_y = 320.0, 320.0, 160.0

        # 각 프레임 정규화 수행
        for t, frame_data in enumerate(core_keypoints):
            if 'keypoints' in frame_data and frame_data['keypoints']:
                keypoints_list = frame_data['keypoints']
                
                if len(keypoints_list) >= 16:
                    # tuple 리스트를 numpy 배열로 변환
                    keypoints = np.array(keypoints_list[:16], dtype=np.float32)
                    
                    normalized_kps = normalize_yolo_pose_keypoints(keypoints, h=h, center_x=center_x, center_y=center_y)
                    
                    if normalized_kps is not None and isinstance(normalized_kps, torch.Tensor):
                        data[t] = normalized_kps.T.numpy()
                    else:
                        print(f"프레임 {t}: 정규화 실패")

        stgcn_tensor = torch.tensor(data, dtype=torch.float32).permute(2, 0, 1)  # (C, T, V)
        
        print(f"ST-GCN 텐서 생성 완료: {stgcn_tensor.shape}")
        # print("Max:", stgcn_tensor.max().item())
        # print("Min:", stgcn_tensor.min().item())
        # print("Mean:", stgcn_tensor.mean().item())
        # print("Std:", stgcn_tensor.std().item())
        
        # 텐서가 모두 0인지 확인
        tensor_sum = torch.sum(torch.abs(stgcn_tensor))
        print(f"텐서 절댓값 합: {tensor_sum.item()}")
        
        if tensor_sum.item() < 1e-6:
            print("❌ 경고: 텐서가 거의 모두 0입니다!")
            return None

        return stgcn_tensor
    
    except Exception as e:
        print(f"텐서 변환 오류: {e}")
        import traceback
        traceback.print_exc()
        return None


# 라우트 정의
@app.route('/')
def index():
    if current_user.is_authenticated:
        recent_videos = VideoAnalysis.query.filter_by(user_id=current_user.id)\
                                         .order_by(VideoAnalysis.upload_time.desc())\
                                         .limit(5).all()
        return render_template('dashboard.html', recent_videos=recent_videos)
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('사용자명이 이미 존재합니다.')
            return render_template('register.html')
            
        if User.query.filter_by(email=email).first():
            flash('이메일이 이미 등록되어 있습니다.')
            return render_template('register.html')
        
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('회원가입이 완료되었습니다. 로그인 해주세요.')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('로그인 성공!')
            return redirect(url_for('index'))
        else:
            flash('사용자 이름 또는 비밀번호가 올바르지 않습니다.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('로그아웃 되었습니다.')
    return redirect(url_for('index'))



@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_video():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('파일이 선택되지 않았습니다.')
            return redirect(request.url)

        file = request.files['video']
        if file.filename == '':
            flash('파일이 선택되지 않았습니다.')
            return redirect(request.url)

        # 사용자가 입력한 저장할 영상 이름 받기
        video_name = request.form.get('video_name', '').strip()
        if not video_name:
            flash('저장할 영상 이름을 입력해주세요.')
            return redirect(request.url)

        if file:
            # 원본 확장자 추출
            ext = os.path.splitext(file.filename)[1]

            # 사용자 입력 이름을 안전하게 처리
            safe_video_name = secure_filename(video_name)
            
            # 만약 secure_filename으로 처리한 결과가 빈 문자열이면 기본값 사용
            if not safe_video_name:
                safe_video_name = "video"

            # 타임스탬프 추가 (중복 방지용)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # 저장할 실제 파일명
            filename = f"{current_user.id}_{timestamp}_{safe_video_name}{ext}"

            # 저장 경로
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # DB에 저장 - stored_filename이 None이 되지 않도록 수정
            video_analysis = VideoAnalysis(
                user_id=current_user.id,
                filename=filename,                    # 저장된 실제 파일명
                original_filename=file.filename,      # 사용자가 업로드한 원본 이름
                stored_filename=video_name,           # 사용자 입력 영상 이름 (safe_video_name 대신 원본 사용)
                upload_time=datetime.now()
            )
            db.session.add(video_analysis)
            db.session.commit()

            return redirect(url_for('analyze_video', video_id=video_analysis.id))

    return render_template('upload.html')


@app.route('/analyze/<int:video_id>')
@login_required
def analyze_video(video_id):
    video = VideoAnalysis.query.filter_by(id=video_id, user_id=current_user.id).first_or_404()
    return render_template('analyze.html', video=video)

@app.route('/api/analyze/<int:video_id>', methods=['POST'])
@login_required
def api_analyze_video(video_id):
    try:
        video = VideoAnalysis.query.filter_by(id=video_id, user_id=current_user.id).first_or_404()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': '비디오 파일을 찾을 수 없습니다.'}), 404
        
        # 비디오 분석 시작 (스켈레톤 영상 생성 포함)
        result = analyze_golf_swing_with_skeleton(filepath)
        
        if result:
            # 결과를 데이터베이스에 저장
            video.prediction_result = result['prediction']
            video.keypoints_data = json.dumps(result['keypoints_data'])
            video.analysis_time = datetime.now()
            video.frame_count = result['frame_count']
            db.session.commit()
            
            return jsonify({
                'success': True,
                'prediction': result['prediction'],
                'frame_count': result['frame_count'],
                'skeleton_video': result.get('skeleton_filename')
            })
        else:
            return jsonify({'error': '분석 중 오류가 발생했습니다.'}), 500
            
    except Exception as e:
        print(f"분석 API 오류: {e}")
        return jsonify({'error': str(e)}), 500

def analyze_golf_swing_with_skeleton(filepath):
    """
    골프 스윙 분석 + 스켈레톤 영상 생성
    
    Args:
        filepath: 원본 영상 경로
        video_name: 사용자 지정 영상 이름 (선택사항)
    """
    try:
        if golf_model.yolo_model is None:
            return None
            
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return None
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 너무 긴 비디오는 제한
        if total_frames > 1000:
            total_frames = 1000

        # 첫번째 프레임에서 bounding box 추출
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return None

        results = golf_model.yolo_model.predict(first_frame, verbose=False)
        keypoints = []
        if results and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            if len(results[0].keypoints.xy) > 0:
                pts = results[0].keypoints.xy[0].cpu().numpy()
                confs = results[0].keypoints.conf[0].cpu().numpy()
                for (x, y), s in zip(pts, confs):
                    keypoints.extend([x, y, score_to_visibility(s)])
        
        bbox = calculate_bbox_from_keypoints(keypoints)
        if bbox is None:
            cap.release()
            return None

        # 전체 프레임 처리
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        keypoints_data = []
        frame_idx = 0
        all_frames = []
        while frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = center_crop_and_resize(frame, bbox)
            all_frames.append(processed_frame)
            result = golf_model.yolo_model.predict(processed_frame, verbose=False)

            frame_keypoints = []
            if result and hasattr(result[0], 'keypoints') and result[0].keypoints is not None:
                if len(result[0].keypoints.xy) > 0:
                    pts = result[0].keypoints.xy[0].cpu().numpy()
                    # 16개 키포인트만 사용
                    for i, (x, y) in enumerate(pts[:16]):
                        frame_keypoints.append((int(x), int(y)))
            
            keypoints_data.append({"frame": frame_idx, "keypoints": frame_keypoints})
            frame_idx += 1

        cap.release()
        
        if len(keypoints_data) == 0:
            return None


        # 기존 분석 로직 계속...
        movements = calculate_left_wrist_mv(keypoints_data)
        prediction = "amateur"  # 기본값
        
        # STGCN 모델 예측 로직은 기존과 동일
        if golf_model.stgcn_model is not None:
            try:
                # 핵심 프레임 추출 로직
                first_target_length = 120
                final_target_length = 80
                
                n = min(20, len(movements))
                top_indices = sorted(range(len(movements)), key=lambda i: movements[i], reverse=True)[:n]

                min_idx = min(top_indices)
                max_idx = max(top_indices)
                core_numbers = max_idx - min_idx + 1

                front_pad = int((first_target_length - core_numbers) / 2)
                back_pad = first_target_length - core_numbers - front_pad

                start_idx = max(0, min_idx - front_pad)
                finish_idx = min(len(keypoints_data) - 1, max_idx + back_pad)

                if finish_idx - start_idx + 1 > first_target_length:
                    finish_idx = start_idx + first_target_length - 1
                elif finish_idx - start_idx + 1 < first_target_length:
                    if start_idx > 0:
                        start_idx = max(0, finish_idx - first_target_length + 1)

                core_keypoints = keypoints_data[start_idx:finish_idx + 1]

                if len(core_keypoints) != first_target_length:
                    core_keypoints = uniform_sample(core_keypoints, first_target_length)

                final_keypoints = uniform_sample(core_keypoints, final_target_length)

                # 스켈레톤 영상 생성
                try:
                    # 출력 파일명 생성
                    filename_only = os.path.basename(filepath)
                    skeleton_filename = draw.get_skeleton_filename(filename_only)

                    
                    # 업로드 폴더에 저장
                    skeleton_output_path = os.path.join(app.config['UPLOAD_FOLDER'], skeleton_filename)
                    
                    frames_only = [all_frames[item['frame']] for item in final_keypoints]

                    keypoints_only = [item['keypoints'] for item in final_keypoints]

                    # 80프레임 스켈레톤 영상 생성
                    skeleton_success = draw.create_skeleton_video_80frames(
                        frames_only,
                        keypoints_only,
                        skeleton_output_path
                    )
                    
                    if skeleton_success:
                        print(f"✅ 스켈레톤 영상 생성 성공: {skeleton_filename}")
                    else:
                        print("❌ 스켈레톤 영상 생성 실패")
                    
                except Exception as e:
                    print(f"❌ 스켈레톤 영상 생성 중 오류: {e}")

                core_tensor = convert_to_stgcn_tensor(final_keypoints)
                
                if core_tensor is not None:
                    input_tensor = core_tensor.unsqueeze(0).to(golf_model.device)
                    
                    with torch.no_grad():
                        output = golf_model.stgcn_model(input_tensor)
                        pred_class = torch.argmax(output, dim=1).item()
                        confidence = torch.softmax(output, dim=1)[0][pred_class].item()

                        print(f"예측 클래스: {CLASS_NAMES[pred_class]}, 신뢰도: {confidence:.4f}")

                        # 조건에 따른 예측 등급 결정
                        if pred_class == 2:
                            prediction = "pro"
                        elif pred_class == 1:
                            prediction = "semipro"
                        else:
                            prediction = "amateur"

            except Exception as e:
                print(f"STGCN 예측 오류: {e}")
        
        return {
            'prediction': prediction,
            'keypoints_data': keypoints_data,
            'frame_count': len(keypoints_data),
            'skeleton_filename': skeleton_filename if 'skeleton_success' in locals() and skeleton_success else None
        }
        
    except Exception as e:
        print(f"골프 스윙 분석 오류: {e}")
        return None

@app.route('/history')
@login_required
def video_history():
    videos = VideoAnalysis.query.filter_by(user_id=current_user.id)\
                               .order_by(VideoAnalysis.upload_time.desc()).all()
    return render_template('history.html', videos=videos)

@app.route('/video/<int:video_id>')
@login_required
def video_detail(video_id):
    video = VideoAnalysis.query.filter_by(id=video_id, user_id=current_user.id).first_or_404()

    # 원본 영상 파일명에서 skeleton_filename 넘기기
    # skeleton 파일 이름 생성
    name, ext = os.path.splitext(video.filename)
    skeleton_filename = f"{name}_skeleton{ext}"

    return render_template('video_detail.html', video=video, skeleton_filename=skeleton_filename)

@app.route('/api/video/<int:video_id>/keypoints')
@login_required
def get_video_keypoints(video_id):
    video = VideoAnalysis.query.filter_by(id=video_id, user_id=current_user.id).first_or_404()
    if video.keypoints_data:
        return jsonify(json.loads(video.keypoints_data))
    return jsonify([])

# 모델 초기화
def initialize_app():
    with app.app_context():
        db.create_all()
        try:
            golf_model.initialize_models()
            print("✅ 모델 초기화 완료")
        except Exception as e:
            print(f"❌ 모델 초기화 실패: {e}")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
