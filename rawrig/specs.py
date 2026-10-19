"""Published specifications, separate from capabilities read from each connected device."""
SPECS = [
    {"model": "BFS-PGE-27S5C", "sensor": "Sony IMX429", "resolution": "1944 × 1472 (현재 RAW ROI 1936 × 1464)",
     "size": "2/3형", "pixel_um": 4.5, "shutter": "Global", "adc_bits": 12,
     "dynamic_range_db": "70.63 / 71.81 (변환 게인 High / Low)", "read_noise_e": "2.63 / 5.79",
     "well_e": "10,652 / 24,495", "qe_green_percent": "60.10 / 61.05 (525 nm)", "snr_max_db": "40.27 / 43.89",
     "interface": "GigE · Spinnaker", "raw": "Bayer16 컨테이너, 유효 ADC 12 bit", "rates": "최대 43 FPS / Bayer16 ISP Off 18 FPS", "controls": "15 µs–30 s / 0–47.9943 dB",
     "source": "https://softwareservices.flir.com/BFS-PGE-27S5/latest/EMVA/EMVA.html"},
    {"model": "BFS-PGE-23S3C", "sensor": "Sony IMX392", "resolution": "1920 × 1200", "size": "1/2.3형",
     "pixel_um": 3.45, "shutter": "Global", "adc_bits": 12, "dynamic_range_db": "71.82",
     "read_noise_e": "2.32", "well_e": "10,987", "qe_green_percent": "59.13 (525 nm)", "snr_max_db": "40.41",
     "interface": "GigE · Spinnaker", "raw": "Bayer16 컨테이너, 유효 ADC 12 bit", "rates": "최대 53 FPS / Bayer16 ADC12 ISP Off 23 FPS", "controls": "23 µs–30 s / 0–47.9943 dB",
     "source": "https://softwareservices.flir.com/BFS-PGE-23S3/latest/EMVA/EMVA.html"},
    {"model": "IMX490 탑재 카메라 · 모델 미확인", "sensor": "Sony IMX490", "resolution": "2896 × 1876 (센서 권장)",
     "size": "1/1.55형", "pixel_um": 3.0, "shutter": "Full pixel reading · 모듈의 동작 모드 확인 필요",
     "adc_bits": "10 / 12", "dynamic_range_db": "120 HDR / 140 DR 우선 모드 (센서 조건)",
     "read_noise_e": "동일 조건 공개 수치 미확인", "well_e": "미확인", "qe_green_percent": "미확인", "snr_max_db": "미확인",
     "interface": "센서 MIPI CSI-2 · 카메라 외부 인터페이스 미확인", "raw": "모듈의 ISP·HDR 합성·선형 RAW 출력 확인 필요",
     "rates": "센서 ADC10 40 FPS / ADC12 30 FPS", "controls": "카메라 모델·HDR 하위 노출시간/게인 확인 필요",
     "source": "https://www.sony-semicon.com/files/62/pdf/p-15_IMX490.pdf"},
    {"model": "Nikon Z5", "sensor": "FX CMOS · 센서 품번 비공개", "resolution": "24.3 MP", "size": "35.9 × 23.9 mm",
     "pixel_um": "약 5.97 (가로 6016 화소 기준 추정)", "shutter": "기계 / 전자 선막 / 전자 · 설정별 차이",
     "adc_bits": "NEF 12 / 14", "dynamic_range_db": "동일 EMVA 조건 수치 미확인", "read_noise_e": "미확인", "well_e": "미확인",
     "qe_green_percent": "미확인", "snr_max_db": "미확인", "interface": "USB-C · gphoto2", "raw": "NEF · ISO 100–51200 · 노출시간 1/8000–30 s",
     "rates": "최대 4.5 FPS (USB 전송 포함 속도 별도)", "controls": "노출시간·ISO·조리개 · 기기 지원 선택지 재확인",
     "source": "https://onlinemanual.nikonimglib.com/z5/en/15_technical_notes_08.html"},
]

def comparison_text():
    lines = ["카메라 특성 비교 · 공개 사양 / 연결 시 실제 설정은 별도 재확인", ""]
    labels = [("sensor", "센서"), ("resolution", "해상도"), ("size", "센서 크기"), ("pixel_um", "화소 크기 µm"),
              ("shutter", "셔터"), ("adc_bits", "ADC/RAW 비트"), ("dynamic_range_db", "DR dB"),
              ("read_noise_e", "읽기 잡음 e−"), ("well_e", "포화 용량 e−"), ("qe_green_percent", "녹색 QE %"),
              ("snr_max_db", "최대 SNR dB"), ("interface", "연결"), ("raw", "RAW"), ("rates", "프레임 속도"), ("controls", "설정 범위")]
    for spec in SPECS:
        lines.append(spec["model"])
        lines.extend(f"  {label}: {spec[key]}" for key, label in labels)
        lines.extend(["  출처: " + spec["source"], ""])
    lines.append("같은 노출시간·게인은 공통 입력 조건 비교에 사용합니다. 센서 감도·렌즈 투과율·조리개·변환 게인·HDR 응답이 달라 같은 밝기나 SNR을 보장하지 않습니다. 각자 자동 측광 모드로 실용 성능도 함께 비교하세요. 포화 비율은 화질 판단용이며 파일 무결성 합격/실패와 별개입니다.")
    return "\n".join(lines)
