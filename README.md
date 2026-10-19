# RAW Rig

GigE 카메라 Bayer RAW 촬영 및 정밀 노이즈 캘리브레이션을 위한 Python / Qt 데스크톱 프로그램입니다.

- 1~3대 라이브 뷰, 설정별 동시 촬영, PTP 예약 Action Command / 소프트웨어 트리거
- RAW 원본 저장, 체크포인트·SHA-256 검증, 중단 후 복구
- 차광 / 밝은 장면 노출시간 순회, 플리커 검사·재촬영, 저조도 분석
- 픽셀별 conversion gain·offset·read noise·추정 오차, 잔차 분포·공분산·PTC
- Nikon Z5 USB 촬영 보조 기능 (선택 사항)

## 설치 및 실행

Linux x86-64에서 개발·검증했습니다. Python 3.10 이상이 필요합니다.

```bash
git clone https://github.com/Choihyunseok1/RAW-Rig.git
cd RAW-Rig
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
./run.sh --demo
```

`--demo`는 가상 카메라입니다. 실제 측정 자료로 사용할 수 없습니다.

### 실제 GigE 카메라

Teledyne FLIR Spinnaker 런타임과 해당 버전의 C 라이브러리를 별도로 설치해야 합니다.
개발 장비는 **Spinnaker 4.4.0.246**을 사용했습니다. SDK는 이 저장소에 포함하지 않습니다.
공식 배포처: <https://www.teledynevisionsolutions.com/products/spinnaker-sdk/>

```bash
# Linux 배포 패키지의 matching libspinnaker-c 파일을 로컬에 추출하거나
./scripts/setup_sdk.sh /path/to/libspinnaker-c_VERSION_amd64.deb
# 설치된 라이브러리를 직접 지정합니다.
export SPINNAKER_C_LIBRARY=/path/to/libSpinnaker_C.so
./run.sh --camera-count 1
```

SpinView 등 다른 프로그램의 카메라 연결을 해제하고, 카메라와 PC 유선 NIC를 같은 서브넷으로 설정합니다.
앱의 **새 카메라 찾기 · 주소 설정**으로 주소·RAW 수신을 검증할 수 있습니다.
PTP 모드는 카메라/네트워크의 PTP 및 Scheduled Action 지원이 필요합니다.
단순 소프트웨어 트리거는 실제 노출 시작의 동시성을 보장하지 않습니다.

### 수치 자료 사용 / 후처리

Slack으로 전달한 `CALIBRATION_NUMBERS_ALL.zip`은 계산된 CSV/NPZ이며 별도 자료입니다.
RAW·실험 결과·SDK·개인별 설정은 Git에 포함하지 않습니다.
수치 ZIP의 `README_KO.txt`, `MODEL_AND_UNITS.json`에서 단위·모델·유효 범위를 확인하세요.

```python
import numpy as np
maps = np.load("CALIBRATION_NUMBERS/pixel_calibration_all_arrays.npz")
gain_dn_per_e = maps["full_gain_dn_per_e"]
valid = maps["full_valid"]
```

전체 촬영 세션(RAW·manifest·통계 파일)이 있을 경우:

```bash
python scripts/analyze_session.py /path/to/noise_session
python scripts/analyze_weighted.py /path/to/noise_session
python scripts/analyze_covariance.py /path/to/noise_session
```

가중 분석과 공분산 스크립트는 **단일 카메라·단일 게인** 세션용입니다.
측정 통계에서 계산하며 RAW 값을 정규화하거나 그래프 모양에 맞춰 수정하지 않습니다.
PTC 전자 축은 측정 conversion gain으로 환산합니다. 입사 광자수와 QE는 광량·스펙트럼 교정 없이 실측으로 간주할 수 없습니다.
공간 패턴에는 조명과 광학계가 포함되고, 경험적 표준오차는 모든 체계오차의 상한이 아닙니다.

## 검증

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 QT_QPA_PLATFORM=offscreen python -m pytest -q
python -m rawrig list --demo
```

상세 설정·촬영·무결성·노이즈 분석 안내: [전체 매뉴얼](docs/manual.md).
