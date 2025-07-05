```
YOLO v8 Pose + ST-gcn
```

import numpy as np
import torch

torch_tensor_save_path = 'swing_sequence.pt'

# 가정: keypoints_list = [np.array((16, 2)) for each frame], len = T_raw

keypoints_list = []  # ← 각 프레임의 결과 append

for frame in video_frames:
    results = model.predict(frame)  # model: YOLO-pose 모델
    if len(results.keypoints.xy) > 0:
        kpts = results.keypoints.xy[0].cpu().numpy()  # [16, 2]
    else:
        kpts = np.zeros((16, 2))  # 감지 실패 시 0
    keypoints_list.append(kpts)

# 현재 프레임 수 (예: 210개) → 120개로 interpolation
T_target = 80
T_raw = len(keypoints_list)
V = 16  # keypoints 수
C = 2

# (T_raw, V, C)
data = np.array(keypoints_list)

# Uniform Sampling (간격 맞추어 샘플링)
# 균등 간격으로 인덱스 선택
indices = np.linspace(0, T_raw - 1, T_target).astype(int)
data_sampled = data[indices]  # (120, 17, 2)

# ST-GCN 포맷으로 변환: [C, T, V, M]
stgcn_input = data_sampled.transpose(2, 0, 1)[..., np.newaxis]  # (2, 120, 17, 1)

print(stgcn_input.shape)  # (2, 120, 16, 1)

# numpy → tensor
tensor_input = torch.tensor(stgcn_input, dtype=torch.float32)

# 저장
torch.save(tensor_input, torch_tensor_save_path)