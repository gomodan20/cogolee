import cv2
import os
import numpy as np
import subprocess


# 16개 관절에 맞는 골프 스켈레톤 연결 정보
GOLF_SKELETON = [
    # 머리-목-가슴 연결
    [0, 1],   # head - neck
    [1, 2],   # neck - chest
    
    # 어깨 연결
    [2, 3],   # chest - right_shoulder
    [2, 4],   # chest - left_shoulder
    
    # 팔 연결
    [3, 5],   # right_shoulder - right_elbow
    [5, 7],   # right_elbow - right_wrist
    [4, 6],   # left_shoulder - left_elbow
    [6, 8],   # left_elbow - left_wrist
    
    # 몸통-골반 연결
    [2, 9],   # chest - hip
    
    # 다리 연결
    [9, 10],  # hip - right_hip
    [9, 11],  # hip - left_hip
    [10, 12], # right_hip - right_knee
    [12, 14], # right_knee - right_ankle
    [11, 13], # left_hip - left_knee
    [13, 15], # left_knee - left_ankle
]

# 키포인트별 색상 (16개)
KEYPOINT_COLORS = [
    (0, 0, 255),     # 0: head - 빨강
    (255, 0, 0),     # 1: neck - 파랑
    (0, 255, 0),     # 2: chest - 초록
    (255, 255, 0),   # 3: right_shoulder - 청록
    (255, 0, 255),   # 4: left_shoulder - 자주
    (128, 255, 0),   # 5: right_elbow - 연두
    (0, 128, 255),   # 6: left_elbow - 주황
    (255, 128, 0),   # 7: right_wrist - 하늘
    (128, 0, 255),   # 8: left_wrist - 보라
    (255, 255, 255), # 9: hip - 흰색
    (128, 128, 128), # 10: right_hip - 회색
    (64, 64, 64),    # 11: left_hip - 진회색
    (255, 192, 203), # 12: right_knee - 분홍
    (255, 165, 0),   # 13: left_knee - 오렌지
    (0, 255, 255),   # 14: right_ankle - 노랑
    (255, 20, 147),  # 15: left_ankle - 딥핑크
]





def draw_keypoints_on_frame(frame, keypoints, skeleton=None, keypoint_threshold=0.3):
    """
    프레임에 키포인트와 스켈레톤을 그리는 함수 (16개 관절용)
    
    Args:
        frame: OpenCV 프레임 (BGR) - numpy array
        keypoints: 키포인트 리스트 [(x1, y1), (x2, y2), ...] - 16개
        skeleton: 스켈레톤 연결 정보 [[joint1, joint2], ...]
        keypoint_threshold: 키포인트 표시 임계값
    """
    if not keypoints or len(keypoints) == 0:
        return frame
    
    # 입력이 numpy array인지 확인
    if not isinstance(frame, np.ndarray):
        print("❌ 프레임이 numpy array가 아닙니다.")
        return frame
    
    frame_copy = frame.copy()
    
    # 스켈레톤 그리기 (선)
    if skeleton:
        for connection in skeleton:
            if len(connection) == 2:
                joint1_idx, joint2_idx = connection
                if (joint1_idx < len(keypoints) and joint2_idx < len(keypoints)):
                    pt1 = keypoints[joint1_idx]
                    pt2 = keypoints[joint2_idx]
                    
                    # 두 점이 모두 유효한 경우에만 선 그리기
                    if (len(pt1) >= 2 and len(pt2) >= 2 and 
                        pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0):
                        try:
                            cv2.line(frame_copy, 
                                    (int(pt1[0]), int(pt1[1])), 
                                    (int(pt2[0]), int(pt2[1])), 
                                    (0, 255, 0), 2)  # 초록색 선, 두께 2
                        except Exception as e:
                            print(f"선 그리기 오류: {e}")
    
    # 키포인트 그리기 (원)
    for i, pt in enumerate(keypoints):
        if len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:  # 유효한 키포인트만
            try:
                color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
                cv2.circle(frame_copy, (int(pt[0]), int(pt[1])), 4, color, -1)  # 원 크기 4
                
                # 키포인트 번호 표시 (선택적)
                cv2.putText(frame_copy, str(i), (int(pt[0]) + 6, int(pt[1]) - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            except Exception as e:
                print(f"키포인트 {i} 그리기 오류: {e}")
    
    return frame_copy


import cv2
import os
import subprocess

def create_skeleton_video_80frames(frames, keypoints_data, output_video_path, fps=20):
    """
    크롭된 프레임들과 키포인트 데이터로 80프레임 스켈레톤 영상을 생성 + 웹 호환 인코딩

    Args:
        frames: 크롭된 프레임들 (numpy arrays)
        keypoints_data: [{'frame': idx, 'keypoints': [(x, y), ...]}, ...]
        output_video_path: 최종 저장 경로 (웹용 mp4)
        fps: 프레임레이트
    """
    try:
        if not frames:
            print("❌ 입력 프레임 없음")
            return False
        if not keypoints_data:
            print("❌ 키포인트 없음")
            return False

        # 임시 파일 경로 설정 (비웹용)
        tmp_output_path = output_video_path.replace(".mp4", "_tmp.mp4")

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            print("❌ VideoWriter 초기화 실패")
            return False

        for i, (frame, keypoints) in enumerate(zip(frames, keypoints_data)):
            try:
                if keypoints and len(keypoints) > 0:
                    keypoints = keypoints[:16]
                    frame_with_skeleton = draw_keypoints_on_frame(frame, keypoints, GOLF_SKELETON)
                else:
                    frame_with_skeleton = frame.copy()

                out.write(frame_with_skeleton)

            except Exception as e:
                print(f"프레임 {i} 처리 오류: {e}")
                out.write(frame)

        out.release()
        print(f"🎬 스켈레톤 영상 저장 완료: {tmp_output_path}")

        # # ✅ FFmpeg로 웹용 재인코딩 (H.264 + yuv420p)
        # ffmpeg_cmd = [
        #     "ffmpeg", "-y",
        #     "-i", tmp_output_path,
        #     "-vcodec", "libx264",
        #     "-pix_fmt", "yuv420p",
        #     "-crf", "23",
        #     "-an",
        #     output_video_path
        # ]
        # subprocess.run(ffmpeg_cmd, check=True)
        # print(f"✅ 웹 호환 스켈레톤 영상 저장 완료: {output_video_path}")

        # # 임시 파일 삭제
        # if os.path.exists(tmp_output_path):
        #     os.remove(tmp_output_path)

        return True

    except Exception as e:
        print(f"❌ 전체 처리 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_skeleton_filename(filename):
    # 확장자가 여러 개 붙은 경우 대비해서 마지막 확장자만 분리
    base, ext = os.path.splitext(filename)
    if ext.lower() not in ['.mp4', '.avi', '.mov']:  # 필요한 확장자만 필터링
        # 확장자가 이상하다면 그냥 붙임
        return filename + "_skeleton"
    return base + "_skeleton" + ext
