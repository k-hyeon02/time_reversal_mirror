from __future__ import annotations

import numpy as np

# ReSpeaker USB 4-Mic Array geometry from the official ODAS config.
# Source: https://raw.githubusercontent.com/respeaker/usb_4_mic_array/master/odas.cfg
# 배열 형태: 정사각형 (한 변 64mm)
RESPEAKER_4CH = np.array(
    [
        [-0.032, 0.000, 0.000],   # 마이크1 (X축 음의 방향, 32mm)
        [0.000, -0.032, 0.000],   # 마이크2 (Y축 음의 방향, 32mm)
        [0.032, 0.000, 0.000],    # 마이크3 (X축 양의 방향, 32mm)
        [0.000, 0.032, 0.000],    # 마이크4 (Y축 양의 방향, 32mm)
    ],
    dtype=np.float32,
)

# Commercial NAO head with 4 microphones. These coordinates are centered so that
# all fixed arrays share the same "array-center" convention used in the paper.
# 배열 형태: 직사각형 모양 (X: ±48mm, Y: ±36mm, Z: 30mm)
NAO_4CH = np.array(
    [
        [0.048, 0.036, 0.030],    # 마이크1 (앞위쪽)
        [0.048, -0.036, 0.030],   # 마이크2 (앞아래쪽)
        [-0.048, 0.036, 0.030],   # 마이크3 (뒤위쪽)
        [-0.048, -0.036, 0.030],  # 마이크4 (뒤아래쪽)
    ],
    dtype=np.float32,
)
# mean(axis=0): 각 채널의 평균값 계산 (4개 마이크 위치의 중심)
# keepdims=True: 차원 유지 (broadcast 호환성)
# -=: 각 마이크 좌표에서 무게중심을 빼서 원점 중심으로 정규화
NAO_4CH -= NAO_4CH.mean(axis=0, keepdims=True)

# LOCATA / EARS robot head with 12 microphones.
# Source: LOCATA final-release documentation, Table 3.
NAO_ROBOT_12CH = np.array(
    [
        [-0.028, 0.030, -0.040],
        [0.006, 0.057, 0.000],
        [0.022, 0.022, -0.046],
        [-0.055, -0.024, -0.025],
        [-0.031, 0.023, 0.042],
        [-0.032, 0.011, 0.046],
        [-0.025, -0.003, 0.051],
        [-0.036, -0.027, 0.038],
        [-0.035, -0.043, 0.025],
        [0.029, -0.048, -0.012],
        [0.034, -0.030, 0.037],
        [0.035, 0.025, 0.039],
    ],
    dtype=np.float32,
)
# 12채널 배열도 원점 중심으로 정규화 (NAO_4CH와 동일)
NAO_ROBOT_12CH -= NAO_ROBOT_12CH.mean(axis=0, keepdims=True)

# 동적으로 생성된 마이크 배열에 작은 무작위 변위(jitter) 추가
# 배열 위치에 현실적인 약간의 불확실성/노이즈 반영
# 데이터 증강: 모델이 정확한 좌표뿐만 아니라 약간의 오차도 견딜 수 있도록 학습
RORIGIN_CM = (-0.5, 0.5) # cm 단위 → -0.005m ~ +0.005m

C_MIN = 4
C_MAX = 12


def get_fixed_array(name: str) -> np.ndarray:  # name으로 mic arrary 불러오기  
    arrays = {
        "respeaker": RESPEAKER_4CH,
        "nao4": NAO_4CH,
        "nao12": NAO_ROBOT_12CH,
    }
    try:
        return arrays[name].copy()
    except KeyError as exc:
        raise ValueError(f"Unknown fixed array '{name}'.") from exc


def pairwise_distance_bounds_cm(num_channels: int) -> tuple[float, float]:
    if not (C_MIN <= num_channels <= C_MAX):
        raise ValueError(
            f"num_channels must be within [{C_MIN}, {C_MAX}], got {num_channels}."
        )

    ratio = (num_channels - C_MIN) / (C_MAX - C_MIN)    # 정규화 비율 - 4채널: 0. 12채널: 1
    r_min = np.random.uniform(max(1.0, 4.0 - 3.0 * ratio), 6.0) # 해당 논문의 Experimental Setup-B 참조
    r_max = np.random.uniform(7.0, max(7.0, 9.0 + 4.0 * ratio))
    return float(r_min), float(r_max)


def _pairwise_distance_bounds_m(
    num_channels: int, rng: np.random.Generator
) -> tuple[float, float]:
    ratio = (num_channels - C_MIN) / (C_MAX - C_MIN)
    r_min = rng.uniform(max(1.0, 4.0 - 3.0 * ratio), 6.0) / 100.0  # 하한 ~ 6cm 사이 난수 -> m로 변환
    r_max = rng.uniform(7.0, max(7.0, 9.0 + 4.0 * ratio)) / 100.0  # 7cm ~ 상한 사이 난수 -> m로 변환
    # 채널이 많을수록 마이크 간 거리를 더 넓게 (r_min ↓, r_max ↑) 배치할 수 있게 설계
    return float(r_min), float(r_max)


# 동적 배열 마이크를 3D 구에 균등하게 배치하기 위해 랜덤 방향 벡터가 필요할 때 사용
def _random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)  # 표준정규분포에서 3개 난수 → [x, y, z] 생성
    norm = np.linalg.norm(vec) + 1e-12  # 벡터의 크기(norm) 계산 (분모가 0이 되는 것을 막기 위해 1e-12 더함)
    return (vec / norm).astype(np.float32)  # 벡터를 크기로 나눠 정규화


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    u1, u2, u3 = rng.random(3)  # [0, 1) 범위의 균일 난수 3개 생성
    # Quaternion 생성 (4개 성분: x, y, z, w): 3D 회전 표현
    q = np.array(
        [
            np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2),
            np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2),
            np.sqrt(u1) * np.sin(2.0 * np.pi * u3),
            np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
        ],
        dtype=np.float32,
    )
    x, y, z, w = q
    # Quaternion을 3×3 회전 행렬로 변환
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def random_rotate(coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    '''
    마이크 배열 좌표를 랜덤 회전
    -------------------------------------------------
    * Input
    coords: 마이크 배열 좌표  ex) 4채널 마이크 - 4*3 matrix
    -------------------------------------------------
    * Output
    coords @ rotation = 회전된 좌표 - 마이크 배열을 다양한 각도로 회전: 모델이 방향에 무관하게 학습하도록 데이터 증강
    '''
    rotation = random_rotation_matrix(rng)
    return (coords @ rotation.T).astype(np.float32)


# 동적으로 생성된 마이크 배열(4~12개)을 3D 공간에 배치하는 함수
def sample_dynamic_array(num_channels: int, 
                         rng: np.random.Generator | None = None, 
                         max_attempts: int = 4096) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()  # 난수 생성기 없으면 새로 생성

    if num_channels < 2:
        raise ValueError("num_channels must be at least 2.")

    r_min, r_max = _pairwise_distance_bounds_m(num_channels, rng)
    max_radius = 0.5 * r_max  # 배치할 구의 반지름

    positions: list[np.ndarray] = []  # 마이크 위치 저장할 list
    for _ in range(num_channels):  # 각 마이크마다
        placed = False
        for _ in range(max_attempts):
            candidate = _random_unit_vector(rng) * rng.uniform(0.0, max_radius)  # random 3D 공간의 한 점

            if not positions:  # 첫 번째 마이크
                positions.append(candidate.astype(np.float32))  # 첫 번째 마이크 랜덤으로 먼저 배치
                placed = True
                break

            # 기존 마이크들과의 거리 계산
            dists = np.linalg.norm(np.stack(positions, axis=0) - candidate, axis=1)
            if np.all(dists >= r_min) and np.all(dists <= r_max):
                positions.append(candidate.astype(np.float32))
                placed = True
                break

        if placed:
            continue

        ring_radius = min(
            max_radius,
            max(r_min / (2.0 * np.sin(np.pi / num_channels) + 1e-6), 0.5 * r_min),
        )
        angle = 2.0 * np.pi * len(positions) / num_channels
        fallback = np.array(
            [
                ring_radius * np.cos(angle),
                ring_radius * np.sin(angle),
                0.0,
            ],
            dtype=np.float32,
        )
        positions.append(fallback)

    coords = np.stack(positions, axis=0)
    coords -= coords.mean(axis=0, keepdims=True)
    coords += rng.uniform(RORIGIN_CM[0], RORIGIN_CM[1], size=coords.shape).astype(
        np.float32
    ) / 100.0
    coords -= coords.mean(axis=0, keepdims=True)
    return coords.astype(np.float32)
