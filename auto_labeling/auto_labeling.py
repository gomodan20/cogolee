import os
import cv2
import json
import re
import argparse
import numpy as np
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "head", "neck", "chest",
    "right_shoulder", "left_shoulder",
    "right_elbow", "left_elbow",
    "right_wrist", "left_wrist",
    "hip", "right_hip", "left_hip",
    "right_knee", "left_knee",
    "right_ankle", "left_ankle"
]

def score_to_visibility(score):
    if score >= 0.6:
        return 2
    elif score >= 0.2:
        return 1
    else:
        return 0

def resize_and_pad(image, target_width=640, target_height=360):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    pad_w, pad_h = target_width - new_w, target_height - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded, scale, left, top

def get_next_prefix(base_folder, prefix_base="legend_swing"):
    os.makedirs(base_folder, exist_ok=True)
    existing_dirs = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    used_numbers = {int(match.group(1)) for d in existing_dirs if (match := re.match(f"{prefix_base}(\\d+)", d))}
    next_number = 1
    while next_number in used_numbers:
        next_number += 1
    return f"{prefix_base}{next_number:02d}"

def clamp(val, minimum, maximum):
    return max(minimum, min(val, maximum))

def calculate_bbox_from_keypoints(keypoints, margin_ratio=0.05, min_margin=5):
    visible_points = [[keypoints[i], keypoints[i+1]] for i in range(0, len(keypoints), 3) if keypoints[i+2] > 0]
    if len(visible_points) < 5:
        return None
    visible_points = np.array(visible_points)
    x_min, y_min = np.min(visible_points, axis=0)
    x_max, y_max = np.max(visible_points, axis=0)
    width, height = x_max - x_min, y_max - y_min
    margin_x = max(width * margin_ratio, min_margin)
    margin_y = max(height * margin_ratio, min_margin)
    x_min = clamp(int(x_min - margin_x), 0, 639)
    y_min = clamp(int(y_min - margin_y), 0, 359)
    x_max = clamp(int(x_max + margin_x), 0, 639)
    y_max = clamp(int(y_max + margin_y), 0, 359)
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def validate_and_clamp_bbox(bbox, img_width, img_height, max_size_ratio=0.8):
    if bbox is None:
        return None
    x, y, w, h = bbox
    x = clamp(int(x), 0, img_width - 1)
    y = clamp(int(y), 0, img_height - 1)
    w = clamp(int(w), 10, img_width - x)
    h = clamp(int(h), 10, img_height - y)
    if w > img_width * max_size_ratio or h > img_height * max_size_ratio:
        return None
    return [x, y, w, h]

def extract_and_process(video_path, model_path, image_path, json_path, visual_path, folder_name):
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(json_path, exist_ok=True)
    os.makedirs(visual_path, exist_ok=True)

    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    over_sample = 120
    step = total_frames / over_sample
    results_all = []

    for i in range(over_sample):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * step))
        ret, frame = cap.read()
        if not ret:
            continue
        padded_img, scale, pad_x, pad_y = resize_and_pad(frame, 640, 360)
        results = model(padded_img)[0]

        keypoints = []
        bbox = None
        right_wrist = left_wrist = None

        if (results.keypoints is not None and results.keypoints.xy is not None and
            results.keypoints.conf is not None and len(results.keypoints.xy) > 0):
            pts = results.keypoints.xy[0].cpu().numpy()
            scores = results.keypoints.conf[0].cpu().numpy()
            for idx, ((x, y), s) in enumerate(zip(pts, scores)):
                padded_x = clamp(int(x), 0, 639)
                padded_y = clamp(int(y), 0, 359)
                keypoints.extend([padded_x, padded_y, score_to_visibility(s)])
                if KEYPOINT_NAMES[idx] == "right_wrist":
                    right_wrist = (padded_x, padded_y)
                if KEYPOINT_NAMES[idx] == "left_wrist":
                    left_wrist = (padded_x, padded_y)
            bbox = calculate_bbox_from_keypoints(keypoints, margin_ratio=0.05, min_margin=5)
            bbox = validate_and_clamp_bbox(bbox, 640, 360)
            results_all.append((frame, keypoints, bbox, right_wrist, left_wrist))

    cap.release()

    if not results_all:
        print("⚠️ 유효한 프레임이 없습니다.")
        return

    wrist_positions = [
        ((rw[0] + lw[0]) / 2, (rw[1] + lw[1]) / 2) if rw and lw else None
        for _, _, _, rw, lw in results_all
    ]

    distances = []
    for i in range(1, len(wrist_positions)):
        if wrist_positions[i] and wrist_positions[i-1]:
            dx = wrist_positions[i][0] - wrist_positions[i-1][0]
            dy = wrist_positions[i][1] - wrist_positions[i-1][1]
            distances.append(np.hypot(dx, dy))
        else:
            distances.append(0)

    cumulative = np.cumsum([0] + distances)
    total_movement = cumulative[-1]

    if total_movement == 0:
        selected = results_all[:80]
    else:
        setup_end = np.searchsorted(cumulative, total_movement * 0.2)
        swing_end = np.searchsorted(cumulative, total_movement * 0.8)
        setup_frames = results_all[:setup_end]
        swing_frames = results_all[setup_end:swing_end]
        finish_frames = results_all[swing_end:]

        def uniform_sample(frames, n):
            if not frames:
                return []
            step = len(frames) / n
            return [frames[int(i * step)] for i in range(n)]

        selected = (
            uniform_sample(setup_frames, 20) +
            uniform_sample(swing_frames, 40) +
            uniform_sample(finish_frames, 20)
        )

    for i, (frame, keypoints, bbox, _, _) in enumerate(selected):
        padded_frame, _, _, _ = resize_and_pad(frame, 640, 360)
        img_filename = f"{folder_name}_{i+1:05d}.jpg"
        img_fullpath = os.path.join(image_path, img_filename)
        cv2.imwrite(img_fullpath, padded_frame)

        json_data = {
            "categories": {"keypoints": KEYPOINT_NAMES},
            "image": {
                "filename": img_filename,
                "copyrighter": "", "date": "", "number": "",
                "action": "", "evaluation": "", "hitting": "",
                "resolution": [640, 360]
            },
            "environment": {"time": "", "location": "", "weather": ""},
            "actor": {"grade": "legend", "sex": "", "age": "", "size": "", "height": ""},
            "usage": {"scope": "", "stage": ""},
            "annotations": []
        }

        if bbox:
            json_data["annotations"].append({"class": "person", "box": bbox})
        json_data["annotations"].append({"class": "person", "points": keypoints})

        with open(os.path.join(json_path, img_filename.replace(".jpg", ".json")), 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        vis_img = padded_frame.copy()
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis_img, "Person", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for idx in range(0, len(keypoints), 3):
            x, y, v = keypoints[idx], keypoints[idx+1], keypoints[idx+2]
            if v > 0:
                cv2.circle(vis_img, (x, y), 4, (0, 255, 0), -1)
                cv2.putText(vis_img, KEYPOINT_NAMES[idx//3], (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imwrite(os.path.join(visual_path, img_filename), vis_img)

    print(f"\n✅ 라벨링 완료: {folder_name}")
    print(f"🔢 처리된 이미지 수: {len(selected)}")

def full_pipeline(video_path, model_path, base_folder, prefix_base="legend_swing"):
    prefix = get_next_prefix(base_folder, prefix_base)
    video_folder = os.path.join(base_folder, prefix)
    image_folder = os.path.join(video_folder, 'image')
    json_folder = os.path.join(video_folder, 'json')
    visual_folder = os.path.join(video_folder, 'visual')

    print(f"[AUTO] 생성된 폴더명: {prefix}")
    extract_and_process(video_path, model_path, image_folder, json_folder, visual_folder, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 기반 골프 스윙 자동 라벨링")
    parser.add_argument("--video", type=str, required=True, help="입력 영상 경로 (.mp4)")
    parser.add_argument("--model", type=str, required=True, help="YOLOv8 모델 경로 (.pt)")
    parser.add_argument("--output", type=str, default="./results", help="기본 저장 폴더")
    parser.add_argument("--prefix", type=str, default="legend_swing", help="폴더 이름 prefix")
    args = parser.parse_args()

    full_pipeline(
        video_path=args.video,
        model_path=args.model,
        base_folder=args.output,
        prefix_base=args.prefix
    )
