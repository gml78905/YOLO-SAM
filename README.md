# yolo_sam

YOLO-World ONNX 객체 탐지와 mobile-SAM 분할을 수행하는 ROS 2 패키지입니다.

## 실행 방법

```bash
cd /URobotics_ws
colcon build --packages-select yolo_sam
source install/setup.bash
ros2 launch yolo_sam yolo_sam.launch.py
```

## 설정 파일

주요 파라미터는 `config/yolo_sam.yaml`에서 설정합니다.

- `onnx_yolo_model_path`: YOLO ONNX 모델 경로
- `onnx_object_list_path`: 클래스 매핑 JSON 경로
- `onnx_sam_encoder_model_path`: MobileSAM encoder ONNX 경로
- `onnx_sam_decoder_model_path`: MobileSAM decoder ONNX 경로
- `object_list`: 최종 publish 대상 라벨 필터 목록
- `yolo_conf`: YOLO 탐지 confidence threshold
- `sam_mask_threshold`: SAM mask threshold (기본 0.0)
- `max_pending_frames`: 처리 대기 프레임 큐 크기 (권장 5)
- `frame_process_period_sec`: 프레임 처리 타이머 주기(초)
- `enable_visualization`: 시각화 이미지 publish on/off (`true`/`false`)

## 참고

- `launch/yolo_sam.launch.py`는 `config/yolo_sam.yaml` 값을 우선 사용하도록 구성되어 있습니다.
- `onnx_object_list_path`는 모델 클래스 매핑에 사용하고, `object_list`는 실제 출력 라벨 필터에 사용합니다.
