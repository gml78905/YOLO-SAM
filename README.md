# yolo_sam

ROS 2 package for YOLO-World ONNX detection and mobile-SAM segmentation.

## Run

```bash
cd /URobotics_ws
colcon build --packages-select yolo_sam
source install/setup.bash
ros2 launch yolo_sam yolo_sam.launch.py
```

## Config

Main parameters are in `config/yolo_sam.yaml`.

- `onnx_model_path`: YOLO ONNX model path
- `onnx_object_list_path`: class mapping JSON path
- `object_list`: labels to publish (filter list)
- `yolo_conf`: detection confidence threshold
- `sam_conf`: SAM confidence threshold
