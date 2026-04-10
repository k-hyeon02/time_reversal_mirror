# Proposal 5: Time-Reversal Mirror + Deep Learning for DOA Estimation

**Author**: 규현 | 경희대학교 물리학과 | 2026.04

**Target**: 물리학과 학술제 (학부생 이해도 고려)

---

## 1. 핵심 물리 원리

파동방정식의 시간 대칭성:

$$\text{파동방정식: } \nabla^2 p = \frac{1}{c^2}\frac{\partial^2 p}{\partial t^2}$$

이 식에 $t$를 $-t$로 바꿔도 형태가 변하지 않는다. 즉, **시간을 거꾸로 돌려도 물리 법칙이 성립**한다.

---

## 2. 한 문장 핵심 아이디어

> "마이크에 녹음된 소리를 시간 역전해서 다시 틀면, 소리가 원래 음원 위치로 되돌아가 모인다. 이 '되돌아가 모이는 패턴'을 DNN에 학습시키면 DOA를 추정할 수 있다."

---

## 3. 직관적 설명 (비유)

### 연못의 파문 역재생

연못에 돌을 던지면 파문이 퍼져나간다. 만약 연못 가장자리에서 물결을 녹화한 뒤, 그 물결을 **반대로 재생**할 수 있다면 — 파문은 다시 돌이 떨어진 그 점으로 수렴한다.

소리도 마찬가지다:

```
[Forward - 음원에서 마이크로]
음원 @ (x0, y0)
  ↓
벽에 반사 → 확산
  ↓
마이크 배열이 수신


[Time-Reversal - 마이크에서 음원으로]
마이크의 녹음을 시간 역전
  ↓
가상으로 역방향 방출
  ↓
음원 원래 위치 (x0, y0)로 수렴 & 포커싱
```

---

## 4. 왜 기존 Time-Reversal Acoustics와 다른가

### 기존 연구 (Mathias Fink, 1990s)

- **Time-Reversal Acoustics**: 물리적으로 스피커 배열에서 time-reversed 신호를 **실제로 재방출**
- 하드웨어 의존, 비용 높음
- 실제 포커싱 달성 (실험실 환경)

### 제안 (이 연구)

- DNN이 time-reversal focusing pattern을 **virtual하게 계산**
- 소프트웨어로 DOA 추정
- 추가 하드웨어 없음, scalable

---

## 5. 수학적 표현

마이크에 녹음된 신호 $p_m(t)$를 시간 역전한 뒤 SRP-PHAT 스타일로 공간에 가상으로 역전파시키면:

**Pseudo-Focusing Map**:

$$\mathcal{F}(\mathbf{r}) = \left| \sum_m \int P_m(\omega) \cdot e^{+j\omega|\mathbf{r}-\mathbf{r}_m|/c} \, d\omega \right|^2$$

- $\mathbf{r}$: 3D 공간 위치
- $P_m(\omega)$: 마이크 $m$의 시간 역전 신호의 푸리에 변환
- $e^{+j\omega|\mathbf{r}-\mathbf{r}_m|/c}$: 역시간 전파 위상
- $|\cdot|^2$: 에너지

**해석**: 이 map의 피크(peak)가 원래 음원 위치에 나타난다.

**DNN의 역할**: 
- 입력: $\mathcal{F}(\mathbf{r})$ (또는 중간 표현)
- 출력: DOA (azimuth, elevation) 또는 3D 위치
- 효과: 노이즈/잔향에 robust한 refinement

---

## 6. 왜 학부생 눈높이에 맞는가

| 포인트 | 설명 |
|--------|------|
| **직관적** | "시간을 되돌리면 소리가 제자리로" — 한 문장으로 설명 가능 |
| **물리 기초 탄탄** | 파동방정식의 $t \to -t$ 대칭성, 모든 물리학과 2~3학년이 배움 |
| **반직관적 결과** | "잔향이 많을수록 더 잘 된다" → 발표에서 청중의 "오!" 순간 |
| **시각화 쉬움** | 3D focusing map을 이미지로 보여주기 좋음 |
| **구현 난이도 적당** | pyroomacoustics + numpy FFT + PyTorch → 학부생도 가능 |
| **물리-AI 교집합** | "음... 이렇게 물리를 써서 AI를 가이드하는 거군?" |

---

## 7. 핵심 매력 포인트

### 7.1 잔향이 "적"에서 "아군"으로

일반적으로 DOA 추정에서:
- 직접음(direct path) ✓ 유용
- 반사음(reverberation) ✗ 방해

하지만 **Time-Reversal**에서는:
- 잔향이 많을수록 → focusing이 더 **tight** (sharp)
- 물리 법칙: 모든 방향에서 온 음파가 원래 음원 위치에서 정확히 간섭 소멸
- 결론: 복잡한 실내환경이 오히려 유리

### 7.2 파동의 본질 드러냄

"음파는 돌이킬 수 없는 화살인가, 아니면 되돌릴 수 있는 파동인가?"

물리학에서:
- 거시적 현상 (소리가 멀어진다): 되돌릴 수 없어 보임
- 미시적 파동 (파동방정식): 완전히 가역적

Time-Reversal은 이 **역설을 수치 실험으로 직접 확인**하게 해준다.

---

## 8. 학술제 발표 플로우

```
[Slide 1] 제목 + "시간을 역전할 수 있을까?"

[Slide 2] 연못의 파문 비유 영상 (GIF)

[Slide 3] 파동방정식: t → -t 대칭

[Slide 4] 음향 상호성 + 시간 대칭 = Time-Reversal Focusing

[Slide 5] 포커싱 맵 시각화 (3D plot)
          - 2D 마이크 배열
          - 45° 음원
          - 포커싱 맵이 정확히 45° 피크 → 청중 "오!"

[Slide 6] 잔향 있을 때 vs 없을 때 비교
          - 직관: 잔향 많음 = 방해
          - 현실: 포커싱이 더 sharp

[Slide 7] DNN + Time-Reversal = Robust DOA

[Slide 8] 실험 결과 + 결론
```

---

## 9. 실험 데모 구상 (PyTorch)

```python
import pyroomacoustics as pra
import numpy as np
import torch

# 1. 방 시뮬레이션
room = pra.ShoeBox([10, 10], fs=16000, materials=pra.Material(0.2), 
                    ray_tracing=True, air_absorption=True)
room.add_microphone([5, 5])  # 마이크 배열
room.add_microphone([5.1, 5])
room.add_source([3, 7])  # 음원

room.simulate()
audio = room.mic_array.signals  # (n_mic, n_sample)

# 2. 시간 역전
audio_reversed = audio[:, ::-1]

# 3. Pseudo-Focusing Map 계산
# (SRP-PHAT 스타일 상관 연산)

# 4. DNN으로 DOA 추정
dnn = SimpleDOANet()
doa_pred = dnn(focusing_map)

# 5. 시각화
plot_focusing_map_3d(focusing_map, true_source=[3, 7])
```

---

## 10. PNCF와의 관계 (상보성)

| 특성 | PNCF | Time-Reversal |
|------|------|---------------|
| 활용하는 물리 | $\phi/\kappa = \mathbf{b}\cdot\mathbf{u}$ (위상 정규화) | $t \to -t$ 대칭 (파동방정식) |
| 잔향 대응 | 가정 깨짐 (약점) | 잔향이 도움 (강점) |
| Feature 수준 | 경량, feature-level | 중량, map-level |
| 구현 복잡도 | 낮음 | 중간 |
| 물리 직관 | 고급 (위상 정규화) | 초급~중급 (시간 대칭) |

**제안**: 환경 적응형 DOA
- 잔향 낮음 → PNCF (물리 가정 성립)
- 잔향 높음 → Time-Reversal focusing (robust)
- Hybrid: 둘 다 계산 후 ensemble

---

## 11. 관련 선행 연구

- **Time-Reversal Acoustics** (Fink et al., 1990s): 하드웨어 기반
- **Beamforming + DNN** (Yang et al., ICASSP 2022): SRP-DNN (이미 리스트에 있음)
- **Focusing maps for DOA** (미미한 연구): 대부분 물리 기반, DNN과 결합한 사례 드뭄

**이 연구의 차별점**:
- Time-Reversal의 물리를 직접 활용한 최초의 DNN pre-processing
- 학부 수준 물리 + 최신 AI의 자연스러운 결합

---

## 12. 학술제 진행 시 Q&A 예상

**Q1**: "이게 기존 비밍포밍(beamforming)과 뭐가 다른데요?"
- **A**: 비밍포밍은 특정 방향으로 focus하는 필터. Time-Reversal은 **모든 공간에 동시에 역파 쏴서 원래 음원점에 자동으로 모인다**. 한 방향이 아니라 3D 맵을 얻음.

**Q2**: "시간을 실제로 역전할 수 있나요?"
- **A**: 수학적으로는 가능. 물리적으로는 코드로 계산. 실제 스피커로 재방출하면 정말 모이는데, 우리는 가상 계산으로 focusing map만 추출.

**Q3**: "잔향이 많으면 더 잘 된다는 게 진짜인가요?"
- **A**: 네. 포커싱 map 결과 보여드릴게요. (3D plot) 잔향이 많을수록 피크가 더 sharp해집니다.

---

## 13. 한 줄 요약

*"파동방정식의 시간 대칭성을 이용해 음원 위치로 자동 수렴하는 포커싱 맵을 계산하고, DNN으로 DOA를 robust하게 추정하는 방법. 잔향이 많을수록 성능이 좋다는 역직관적 결과가 특징."*

---

## 14. 발표 팁

- ✓ 연못 물결 영상/GIF 준비 (생성 AI 또는 시뮬레이션)
- ✓ 3D 포커싱 맵 애니메이션 (회전하면서 보이기)
- ✓ "시간을 거꾸로 돌려도 물리 법칙은 같다" — 이 문장에 5초 정지
- ✓ 잔향 비교 before/after (한눈에 차이 보이기)
- ✓ 깃허브 링크 또는 QR 코드 (코드 공개)

---

**이 제안이 학술제에서 빛날 이유**:
1. 물리 기초 탄탄 (학과 정체성)
2. 결과가 반직관적 (청중 주목)
3. 시각화 예쁨 (포스터, 발표 자료)
4. 최신 AI와 고전 물리의 조화 (교수진 호평)
5. 구현 난이도 적당 (학부생 credibility)
