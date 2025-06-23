import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import math

class ConvTemporalGraphical(nn.Module):
    """
    시간적, 공간적 그래프 컨볼루션 레이어
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))

        return x.contiguous()

class st_gcn(nn.Module):
    """
    ST-GCN 블록을 구현합니다.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class Model(nn.Module):
    """
    골프 스윙 분류를 위한 ST-GCN 모델
    """
    def __init__(self, 
                 in_channels, # 입력 keypoints의 채널 수(2차원이면 2)
                 num_class, # 분류할 클래스 수
                 graph_args, # 그래프 구조 정의 (노드 연결 관계 등)
                 edge_importance_weighting=True, # 에지(관절 연결선)에 가중치를 학습할지 여부
                 **kwargs):
        super().__init__()

        # 그래프 로드: 관절간의 연결구조 정의. A는 인접행렬 ([K, V, V]) 형태.
        # K: edge type 수 (보통 1), V: 관절수()
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # ST-GCN 네트워크 구축
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9  # 시간적 커널 크기
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        
        # 배치 정규화 
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        
        # ST-GCN 레이어 스택
        # 각 st_gcn은 Spatial GCN + Temporal Convolution으로 구성된 블록
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 64, kernel_size, 1),
            st_gcn(64, 128, kernel_size, 2),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 128, kernel_size, 1),
            st_gcn(128, 256, kernel_size, 2),
            st_gcn(256, 256, kernel_size, 1),
            st_gcn(256, 256, kernel_size, 1),
        ))

        # 에지 가중치 초기화, 학습
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in range(len(self.st_gcn_networks))
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # FCN 분류기
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
        # ST-GCN의 마지막 출력을 클래스 점수로 변환합니다. 256 차원의 피처를 num_class 차원의 로짓 (logit) 으로 매핑

    def forward(self, x):
        # 데이터 포맷: N, C, T, V
        N, C, T, V = x.size()

        # 데이터 정규화
        x = x.permute(0, 3, 1, 2).contiguous()  # N, V, C, T
        x = x.view(N, V * C, T)
        x = self.data_bn(x)    #  배치 정규화
        x = x.view(N, V, C, T) # 원상복구
        x = x.permute(0, 2, 3, 1).contiguous()  # N, C, T, V
        
        # 순전파 (각 ST-GCN 블록을 통과하며 특징을 추출)
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)

        # 전역 평균 풀링: 시간과 관절 정보를 모두 요약하여 고수준 특징만 남김
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        # 예측 (마지막 특징을 클래스 수로 변환)
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        return x

class Graph():
    """
    골격 그래프를 정의하는 클래스
    """
    def __init__(self, layout='golf', strategy='spatial', max_hop=1, dilation=1):
        # layout: 그래프 설계 방식 (여기선 'golf'로 고정)
        # strategy: adjacency matrix 계산 전략 (여기선 'spatial')
        # max_hop: hop 거리의 최대값 (몇 단계까지 연결을 볼지) => 1 또는 2가 일반적이다. 1로 시작해서 2로 늘려보기
        # dilation: hop 거리의 간격
        # get_edge: 관절과 간선 정의
        # get_hop_distance: 노드 간 최단거리(=hop distance) 계산 (함수 외부 정의 필요)
        # get_adjacency: adjacency matrix 생성 (아래에 상세)

        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        # 골프 스윙 골격 그래프에 맞게 조정
        if layout == 'golf':
            self.num_node = 16  # 16개 관절
            self_link = [(i, i) for i in range(self.num_node)]
            
            # 골프 스윙 관절 연결 그래프 정의
            # 실제 데이터셋에 맞게 수정 필요
            neighbor_link = [
                (0, 1),   # head - neck
                (1, 2),   # neck - chest
                (2, 3),   # chest - right_shoulder
                (2, 4),   # chest - left_shoulder
                (3, 5),   # right_shoulder - right_elbow
                (4, 6),   # left_shoulder - left_elbow
                (5, 7),   # right_elbow - right_wrist
                (6, 8),   # left_elbow - left_wrist
                (2, 9),   # chest - hip
                (9, 10),  # hip - right_hip
                (9, 11),  # hip - left_hip
                (10, 12), # right_hip - right_knee
                (11, 13), # left_hip - left_knee
                (12, 14), # right_knee - right_ankle
                (13, 15), # left_knee - left_ankle
            ]
            self.edge = self_link + [(j, i) for (i, j) in neighbor_link]
            # 자기 자신으로 가는 edge (feature 보존 위해) + 해부학적 연결을 양방향 쌍으로 지정
            self.center = 2  # chest를 중심으로 설정. neck이나 hip은 움직임의 중심 축으로서 의미가 있을 수 있음
            
        else:
            raise ValueError("Unknown layout: {}".format(layout))

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation) # hop 단위로 노드 간 연결 정의
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop: # self.hop_dis는 노드간 최소 거리 행렬
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency) # 인접 행렬 정규화 (예: 행 합이 1이 되도록)

        if strategy == 'spatial':
            # spatial 전략은 ST-GCN 논문에서 제안된 방법으로, adjacency matrix를 3가지 파트로 나눕니다:
            # a_root: 기준 노드와 같은 거리의 노드 연결 (자기 자신 포함)
            # a_close: 중심 노드에서 더 가까운 노드 쪽 연결
            # a_further: 중심 노드에서 더 먼 노드 쪽 연결
            # 이렇게 나눠서 그래프를 여러 adjacency matrix 채널로 분리해 공간적 관계를 더 정교하게 학습하게 합니다.
            # 마지막에 np.stack(A)로 여러 adjacency matrix를 하나의 3차원 텐서로 만들어 self.A에 저장
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.extend([a_root, a_close, a_further])
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("Unknown strategy: {}".format(strategy))

#########################################################################
def get_hop_distance(num_node, edge, max_hop=1):
    # 그래프의 각 노드 쌍 간의 최단 경로(hop) 거리를 계산하는 함수
    # 그래프 내 노드 개수, 노드 간 연결정보, 계산할 최대 거리를 입력
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # 최단 거리 계산
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    # 방향 그래프 (Directed graph) 의 인접 행렬 A를 정규화
    # 입력은 A: 방향 그래프의 인접 행렬 (노드 수 x 노드 수)
    
    Dl = np.sum(A, 0) # 각 열의 합 Dl 계산 (출발 노드 기준 간선 개수)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node): # Dl의 역수(diagonal matrix Dn)로 행렬 곱하여 각 노드에서 나가는 간선이 합이 1이 되도록 정규화
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn) # 정규화된 인접 행렬 AD를 출력
    return AD


def normalize_undigraph(A):
    # 무방향 그래프 (Undirected graph) 의 인접 행렬 A를 정규화
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)  # Dl의 -0.5 제곱(= 역수의 제곱근)을 대각선으로 하는 행렬 Dn 생성
    DAD = np.dot(np.dot(Dn, A), Dn) # Dn * A * Dn 형태로 행렬을 정규화 (대칭 정규화)
    return DAD  # 대칭적으로 정규화된 인접 행렬 DAD를 반환
