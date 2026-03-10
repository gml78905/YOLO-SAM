import os
import json
import time
import base64
from collections import deque
from typing import List, Set

import cv2
import numpy as np
import onnxruntime as ort
import rclpy
import supervision as sv
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    # Optional, for better NMS when ONNX is exported without NMS
    from torchvision.ops import nms, batched_nms
except Exception:
    nms = None
    batched_nms = None

from scene_graph.utils.config import build_config

PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
SAM_IMAGE_SIZE = 1024


class YoloWorldOnnxNode(Node):
    """
    ROS2 node:
      - subscribes to RGB image topic
      - runs YOLO-World ONNX (/data/yolo-world_custom_5cls.onnx) via ONNXRuntime GPU
      - runs mobile-SAM for masks
      - publishes annotated image
    """

    def __init__(self) -> None:
        super().__init__("yolo_sam_node")

        self.get_logger().info("[YOLO-ONNX] Building config...")
        cfg = build_config()
        self.cfg = cfg

        image_topic = self.declare_parameter(
            "image_topic", cfg.image_topic
        ).get_parameter_value().string_value
        viz_topic = self.declare_parameter(
            "visualization_topic", "visualization/test"
        ).get_parameter_value().string_value
        det_topic = self.declare_parameter(
            "detections_topic", "yolo_sam/detections"
        ).get_parameter_value().string_value

        self.get_logger().info(
            f"[YOLO-ONNX] Subscribing image from: {image_topic}, "
            f"publishing visualization to: {viz_topic}, detections(+segments) to: {det_topic}"
        )

        self.bridge = CvBridge()

        # Model/resource paths and detector options
        onnx_yolo_model_path = self.declare_parameter(
            "onnx_yolo_model_path", "/data/yolo-world_custom_5cls.onnx"
        ).get_parameter_value().string_value
        onnx_sam_encoder_model_path = self.declare_parameter(
            "onnx_sam_encoder_model_path", "/data/mobile_sam_encoder.onnx"
        ).get_parameter_value().string_value
        onnx_sam_decoder_model_path = self.declare_parameter(
            "onnx_sam_decoder_model_path", "/data/mobile_sam_decoder.onnx"
        ).get_parameter_value().string_value
        self.sam_mask_threshold = self.declare_parameter(
            "sam_mask_threshold", 0.0
        ).get_parameter_value().double_value
        onnx_object_list_path = self.declare_parameter(
            "onnx_object_list_path", ""
        ).get_parameter_value().string_value
        object_list = self.declare_parameter(
            "object_list", str(getattr(cfg, "object_classes", "sign,bench,car,building,tower,firehydrant"))
        ).get_parameter_value().string_value

        # YOLO thresholds from ROS params (YAML configurable)
        self.yolo_score_thr = self.declare_parameter(
            "yolo_conf", float(getattr(cfg, "yolo_conf", 0.30))
        ).get_parameter_value().double_value
        self.yolo_topk = self.declare_parameter(
            "yolo_topk", int(getattr(cfg, "yolo_topk", 100))
        ).get_parameter_value().integer_value
        self.nms_iou_threshold = self.declare_parameter(
            "nms_iou_threshold", 0.7
        ).get_parameter_value().double_value
        self.max_pending_frames = int(
            self.declare_parameter("max_pending_frames", 5).get_parameter_value().integer_value
        )
        if self.max_pending_frames < 1:
            self.max_pending_frames = 1
        self.frame_process_period_sec = float(
            self.declare_parameter("frame_process_period_sec", 0.001)
            .get_parameter_value()
            .double_value
        )
        if self.frame_process_period_sec <= 0.0:
            self.frame_process_period_sec = 0.001
        self.enable_visualization = self.declare_parameter(
            "enable_visualization", True
        ).get_parameter_value().bool_value

        # ------------------------------------------------------------------
        # Prepare YOLO-World class texts from config
        # ------------------------------------------------------------------
        self.yolo_texts: List[List[str]] = []
        try:
            class_map_source = onnx_object_list_path if onnx_object_list_path else object_list
            if isinstance(class_map_source, str) and os.path.isfile(class_map_source):
                if class_map_source.endswith(".txt"):
                    with open(class_map_source, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    self.yolo_texts = [[t.strip()] for t in lines if t.strip()]
                elif class_map_source.endswith(".json"):
                    with open(class_map_source, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    labels: List[str] = []
                    if isinstance(data, list):
                        for item in data:
                            # Only supports [["sign"], ["tower"], ...]
                            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], str) and item[0].strip():
                                labels.append(item[0].strip())

                    self.yolo_texts = [[name] for name in labels]
                else:
                    # Fallback: text file with one class per line
                    with open(class_map_source, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    self.yolo_texts = [[t.strip()] for t in lines if t.strip()]
            elif isinstance(class_map_source, str):
                self.yolo_texts = [[t.strip()] for t in class_map_source.split(",") if t.strip()]
            else:
                self.yolo_texts = [[str(t)] for t in list(class_map_source)]
        except Exception as e:
            self.get_logger().warning(
                f"[YOLO-ONNX] Failed to parse object_classes: {e}"
            )
            self.yolo_texts = []

        # object_list is used as detection filter list.
        self.allowed_object_labels: Set[str] = set()
        try:
            if isinstance(object_list, str) and os.path.isfile(object_list):
                if object_list.endswith(".txt"):
                    with open(object_list, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    self.allowed_object_labels = {
                        t.strip().lower() for t in lines if isinstance(t, str) and t.strip()
                    }
                elif object_list.endswith(".json"):
                    with open(object_list, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    labels: List[str] = []
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], str):
                                if item[0].strip():
                                    labels.append(item[0].strip())
                    self.allowed_object_labels = {t.lower() for t in labels if t}
                else:
                    with open(object_list, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    self.allowed_object_labels = {
                        t.strip().lower() for t in lines if isinstance(t, str) and t.strip()
                    }
            elif isinstance(object_list, str):
                self.allowed_object_labels = {
                    t.strip().lower() for t in object_list.split(",") if t.strip()
                }
        except Exception as e:
            self.get_logger().warning(
                f"[YOLO-ONNX] Failed to parse object_list filter: {e}"
            )
            self.allowed_object_labels = set()

        self.allowed_class_ids: Set[int] = set()
        if self.allowed_object_labels:
            for idx, names in enumerate(self.yolo_texts):
                if not names:
                    continue
                label = str(names[0]).strip().lower()
                if label in self.allowed_object_labels:
                    self.allowed_class_ids.add(idx)

        # ------------------------------------------------------------------
        # Initialize SAM ONNX encoder/decoder sessions
        # ------------------------------------------------------------------
        self.sam_encoder_session = None
        self.sam_decoder_session = None
        try:
            available = ort.get_available_providers()
            providers: List[str] = []
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
                if p in available:
                    providers.append(p)
            self.get_logger().info(
                f"[YOLO-ONNX] Initializing SAM ONNX encoder/decoder with providers: {providers}"
            )
            self.sam_encoder_session = ort.InferenceSession(
                onnx_sam_encoder_model_path, providers=providers
            )
            self.sam_decoder_session = ort.InferenceSession(
                onnx_sam_decoder_model_path, providers=providers
            )
            self.get_logger().info("[YOLO-ONNX] SAM ONNX sessions initialized successfully.")
        except Exception as e:
            self.sam_encoder_session = None
            self.sam_decoder_session = None
            self.get_logger().warning(f"[YOLO-ONNX] SAM ONNX setup failed: {e}")

        # ------------------------------------------------------------------
        # Initialize YOLO-World ONNX Runtime session (GPU)
        # ------------------------------------------------------------------
        self.onnx_session = None
        self.onnx_input_name: str = "images"
        # ONNX output 구성을 동적으로 판별 (with / without NMS)
        self.onnx_output_names = ["num_dets", "labels", "scores", "boxes"]
        self.onnx_image_size = (640, 640)
        self.onnx_has_nms: bool = True

        try:
            onnx_path = onnx_yolo_model_path
            self.get_logger().info(f"[YOLO-ONNX] Initializing ONNX from: {onnx_path}")

            # Prefer pure ORT CUDA (TensorRT EP는 cuDNN 의존성 때문에 비활성화)
            available = ort.get_available_providers()
            providers: List[str] = []
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
                if p in available:
                    providers.append(p)

            self.get_logger().info(f"[YOLO-ONNX] Using ORT providers: {providers}")
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)

            # Infer input / output names from model
            try:
                inputs = self.onnx_session.get_inputs()
                if inputs:
                    self.onnx_input_name = inputs[0].name
                output_names = [o.name for o in self.onnx_session.get_outputs()]
                self.get_logger().info(f"[YOLO-ONNX] ONNX outputs: {output_names}")

                # Case 1: exported with NMS (num_dets, labels, scores, boxes)
                if all(n in output_names for n in ["num_dets", "labels", "scores", "boxes"]):
                    self.onnx_has_nms = True
                    self.onnx_output_names = ["num_dets", "labels", "scores", "boxes"]
                    self.get_logger().info("[YOLO-ONNX] Detected ONNX with built-in NMS.")
                # Case 2: exported without NMS (scores, boxes)
                elif all(n in output_names for n in ["scores", "boxes"]):
                    self.onnx_has_nms = False
                    self.onnx_output_names = ["scores", "boxes"]
                    self.get_logger().info("[YOLO-ONNX] Detected ONNX without NMS (scores, boxes).")
                else:
                    self.get_logger().warning(
                        f"[YOLO-ONNX] Unexpected ONNX outputs: {output_names} "
                        "(assuming no-NMS: scores, boxes)."
                    )
                    self.onnx_has_nms = False
                    # best-effort: try to use first 2 outputs
                    if len(output_names) >= 2:
                        self.onnx_output_names = output_names[:2]
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] Failed to inspect ONNX IO: {e}")

            self.get_logger().info("[YOLO-ONNX] ONNX Runtime session initialized.")
        except Exception as e:
            self.get_logger().error(f"[YOLO-ONNX] Failed to initialize ONNX session: {e}")
            self.onnx_session = None

        # Frame queue (bounded): keep at most N latest frames.
        self.frame_queue = deque(maxlen=self.max_pending_frames)
        self.dropped_frame_count = 0

        # ROS interfaces
        sub_queue_depth = max(1, self.max_pending_frames)
        self.image_sub = self.create_subscription(
            Image, image_topic, self.image_callback, sub_queue_depth
        )
        self.viz_pub = self.create_publisher(Image, viz_topic, 10)
        self.det_pub = self.create_publisher(String, det_topic, 10)
        self.process_timer = self.create_timer(
            self.frame_process_period_sec, self.process_next_frame
        )

        # Optional tracker
        try:
            self.tracker = sv.ByteTrack()
            self.get_logger().info("[YOLO-ONNX] ByteTrack initialized.")
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] ByteTrack disabled: {e}")
            self.tracker = None

    # ------------------------------------------------------------------
    # Image callbacks and processing loop
    # ------------------------------------------------------------------
    def image_callback(self, msg: Image) -> None:
        # Bounded queue: drop oldest frame when full.
        if len(self.frame_queue) >= self.max_pending_frames:
            self.frame_queue.popleft()
            self.dropped_frame_count += 1
        self.frame_queue.append(msg)

    def process_next_frame(self) -> None:
        if not self.frame_queue:
            return
        msg = self.frame_queue.popleft()
        self._process_image_msg(msg)

    def _process_image_msg(self, msg: Image) -> None:
        t_start = time.perf_counter()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] Failed to convert image: {e}")
            return

        if self.onnx_session is None:
            return

        image = cv_image.copy()

        # --- YOLO-World ONNX inference ---
        t_yolo_start = time.perf_counter()
        try:
            dets = self._run_yolo_onnx(image)
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] ONNX inference failed: {e}")
            dets = []
        t_yolo_end = time.perf_counter()
        yolo_ms = (t_yolo_end - t_yolo_start) * 1000.0

        # --- Tracking (ByteTrack) ---
        tracker_ms = 0.0
        tracked_dets = dets
        if self.tracker is not None and len(dets) > 0:
            try:
                t_tracker_start = time.perf_counter()
                det_xyxy = np.asarray([d["bbox"] for d in dets], dtype=np.float32)
                det_cls = np.asarray([int(d.get("class_id", -1)) for d in dets], dtype=np.int32)
                det_conf = np.asarray([float(d.get("score", 0.0)) for d in dets], dtype=np.float32)
                detections_sv = sv.Detections(
                    xyxy=det_xyxy,
                    class_id=det_cls,
                    confidence=det_conf,
                )
                tracks = self.tracker.update_with_detections(detections_sv)
                tracked_dets = []
                for i in range(len(tracks)):
                    cid = int(tracks.class_id[i]) if tracks.class_id is not None else -1
                    score = float(tracks.confidence[i]) if tracks.confidence is not None else 0.0
                    tid = -1
                    if getattr(tracks, "tracker_id", None) is not None and i < len(tracks.tracker_id):
                        if tracks.tracker_id[i] is not None:
                            tid = int(tracks.tracker_id[i])
                    label = str(cid)
                    if 0 <= cid < len(self.yolo_texts):
                        label = self.yolo_texts[cid][0]
                    tracked_dets.append(
                        {
                            "label": label,
                            "score": score,
                            "bbox": tracks.xyxy[i].tolist(),
                            "class_id": cid,
                            "track_id": tid,
                        }
                    )
                t_tracker_end = time.perf_counter()
                tracker_ms = (t_tracker_end - t_tracker_start) * 1000.0
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] ByteTrack failed: {e}")

        # Run SAM for masks
        sam_masks_np = None
        sam_ms = 0.0
        if (
            self.sam_encoder_session is not None
            and self.sam_decoder_session is not None
            and len(tracked_dets) > 0
        ):
            try:
                sam_masks_np, sam_ms = self._run_sam_onnx(image, tracked_dets)
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] SAM inference failed: {e}")

        if self.enable_visualization:
            # Draw masks (if any), then boxes and labels
            for idx, det in enumerate(tracked_dets):
                bbox = det.get("bbox")
                label = det.get("label", "")
                score = det.get("score", 0.0)
                track_id = det.get("track_id", -1)
                if bbox is None or len(bbox) != 4:
                    continue

                if sam_masks_np is not None and idx < sam_masks_np.shape[0]:
                    try:
                        mask = sam_masks_np[idx]
                        if mask is not None:
                            mask_bool = mask.astype(bool)
                            if mask_bool.shape[:2] == image.shape[:2]:
                                color = np.array([0, 0, 255], dtype=np.uint8)
                                alpha = 0.5
                                image[mask_bool] = (
                                    (1.0 - alpha) * image[mask_bool] + alpha * color
                                ).astype(np.uint8)
                    except Exception as e:
                        self.get_logger().warning(
                            f"[YOLO-ONNX] SAM mask overlay failed: {e}"
                        )

                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if track_id is not None and track_id >= 0:
                    text = f"{label} {score:.2f} ID:{track_id}"
                else:
                    text = f"{label} {score:.2f}"
                cv2.putText(
                    image,
                    text,
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            try:
                img_msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
            except Exception as e:
                self.get_logger().warning(
                    f"[YOLO-ONNX] Failed to convert annotated image: {e}"
                )
                return

            now = self.get_clock().now().to_msg()
            img_msg.header.stamp = now
            img_msg.header.frame_id = "camera_color_optical_frame"
            self.viz_pub.publish(img_msg)

        # Attach segmentation payload directly into detections (single-section JSON).
        detections_payload = [dict(d) for d in tracked_dets]
        if sam_masks_np is not None:
            try:
                for idx, det in enumerate(detections_payload):
                    if idx >= sam_masks_np.shape[0]:
                        continue
                    mask = sam_masks_np[idx]
                    if mask is None:
                        continue

                    mask_u8 = (mask.astype(np.uint8) * 255)
                    ok, encoded = cv2.imencode(".png", mask_u8)
                    if not ok:
                        continue

                    det["mask_png_base64"] = base64.b64encode(encoded.tobytes()).decode("ascii")
                    det["mask_height"] = int(mask_u8.shape[0])
                    det["mask_width"] = int(mask_u8.shape[1])
            except Exception as e:
                self.get_logger().warning(f"[YOLO-ONNX] Failed to encode per-detection masks: {e}")

        # Single-section JSON publish: all info is inside detections[].
        try:
            det_msg = String()
            det_msg.data = json.dumps(
                {
                    # Preserve camera source header stamp as-is.
                    "header": {
                        "stamp": {
                            "sec": int(msg.header.stamp.sec),
                            "nanosec": int(msg.header.stamp.nanosec),
                        },
                        "frame_id": msg.header.frame_id,
                    },
                    "detections": detections_payload,
                },
                ensure_ascii=False,
            )
            self.det_pub.publish(det_msg)
        except Exception as e:
            self.get_logger().warning(f"[YOLO-ONNX] Failed to publish detections: {e}")

        # --- Timing log ---
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0
        self.get_logger().info(
            f"[YOLO-ONNX] Timing: yolo={yolo_ms:.1f}ms, tracker={tracker_ms:.1f}ms, "
            f"sam={sam_ms:.1f}ms, total={total_ms:.1f}ms, "
            f"queue={len(self.frame_queue)}/{self.max_pending_frames}, dropped={self.dropped_frame_count}"
        )

    # ------------------------------------------------------------------
    # YOLO-World ONNX helpers
    # ------------------------------------------------------------------
    def _preprocess_for_onnx(self, image_bgr: np.ndarray):
        size = self.onnx_image_size
        h, w = image_bgr.shape[:2]
        max_size = max(h, w)
        scale_factor = size[0] / max_size
        pad_h = (max_size - h) // 2
        pad_w = (max_size - w) // 2

        pad_image = np.zeros((max_size, max_size, 3), dtype=image_bgr.dtype)
        # BGR -> RGB
        pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image_bgr[:, :, [2, 1, 0]]

        image = cv2.resize(
            pad_image,
            size,
            interpolation=cv2.INTER_LINEAR,
        ).astype("float32")
        image /= 255.0
        image = image[None]  # (1, H, W, 3)
        return image, scale_factor, (pad_h, pad_w)

    def _preprocess_for_sam_encoder(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = SAM_IMAGE_SIZE / max(h, w)
        new_h = int(h * scale + 0.5)
        new_w = int(w * scale + 0.5)

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        resized = resized.astype(np.float32)
        resized = (resized - PIXEL_MEAN) / PIXEL_STD

        padded = np.zeros((SAM_IMAGE_SIZE, SAM_IMAGE_SIZE, 3), dtype=np.float32)
        padded[:new_h, :new_w, :] = resized
        chw = np.transpose(padded, (2, 0, 1))[None, ...]
        return chw, (new_h, new_w), (h, w)

    def _transform_boxes_xyxy_for_sam(
        self, boxes_xyxy: np.ndarray, original_size: tuple[int, int]
    ) -> np.ndarray:
        old_h, old_w = original_size
        scale = SAM_IMAGE_SIZE / max(old_h, old_w)
        new_h = int(old_h * scale + 0.5)
        new_w = int(old_w * scale + 0.5)

        boxes = boxes_xyxy.astype(np.float32).copy()
        boxes[:, [0, 2]] *= float(new_w) / float(old_w)
        boxes[:, [1, 3]] *= float(new_h) / float(old_h)
        return boxes

    def _decode_sam_masks_for_boxes(
        self,
        image_embedding: np.ndarray,
        boxes_xyxy_model_frame: np.ndarray,
        orig_hw: tuple[int, int],
    ) -> np.ndarray:
        if self.sam_decoder_session is None:
            raise RuntimeError("SAM decoder session is not initialized.")

        num_boxes = int(boxes_xyxy_model_frame.shape[0])
        if num_boxes == 0:
            return np.zeros((0, orig_hw[0], orig_hw[1]), dtype=np.float32)

        decoder_inputs_meta = {i.name: i for i in self.sam_decoder_session.get_inputs()}
        point_labels_meta = decoder_inputs_meta.get("point_labels")
        orig_meta = decoder_inputs_meta.get("orig_im_size")

        point_labels_first_dim = None
        try:
            if point_labels_meta is not None and len(point_labels_meta.shape) >= 1:
                first_dim = point_labels_meta.shape[0]
                if isinstance(first_dim, int):
                    point_labels_first_dim = first_dim
        except Exception:
            pass

        def _orig_im_size_for_batch(batch_n: int) -> np.ndarray:
            base = np.array([orig_hw[0], orig_hw[1]], dtype=np.float32)
            try:
                if orig_meta is not None and len(orig_meta.shape) == 2:
                    return np.tile(base[None, :], (batch_n, 1))
            except Exception:
                pass
            return base

        # If decoder export fixed batch to 1, fallback to per-box decode.
        if point_labels_first_dim == 1 and num_boxes > 1:
            masks_out = []
            for i in range(num_boxes):
                box = boxes_xyxy_model_frame[i]
                point_coords = np.array(
                    [[[box[0], box[1]], [box[2], box[3]]]], dtype=np.float32
                )
                point_labels = np.array([[2.0, 3.0]], dtype=np.float32)
                decoder_inputs = {
                    "image_embeddings": image_embedding.astype(np.float32),
                    "point_coords": point_coords,
                    "point_labels": point_labels,
                    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input": np.zeros((1,), dtype=np.float32),
                    "orig_im_size": _orig_im_size_for_batch(1),
                }
                masks, _, _ = self.sam_decoder_session.run(None, decoder_inputs)
                masks_out.append(masks[0, 0])
            return np.stack(masks_out, axis=0)

        # Decoder supports batched prompts.
        point_coords = np.zeros((num_boxes, 2, 2), dtype=np.float32)
        point_coords[:, 0, 0] = boxes_xyxy_model_frame[:, 0]
        point_coords[:, 0, 1] = boxes_xyxy_model_frame[:, 1]
        point_coords[:, 1, 0] = boxes_xyxy_model_frame[:, 2]
        point_coords[:, 1, 1] = boxes_xyxy_model_frame[:, 3]
        point_labels = np.tile(np.array([[2.0, 3.0]], dtype=np.float32), (num_boxes, 1))
        image_embeddings = np.repeat(image_embedding.astype(np.float32), num_boxes, axis=0)

        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "mask_input": np.zeros((num_boxes, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros((num_boxes,), dtype=np.float32),
            "orig_im_size": _orig_im_size_for_batch(num_boxes),
        }
        masks, _, _ = self.sam_decoder_session.run(None, decoder_inputs)
        return masks[:, 0, :, :]

    def _run_sam_onnx(self, image_bgr: np.ndarray, dets: List[dict]):
        if self.sam_encoder_session is None or self.sam_decoder_session is None:
            return None, 0.0

        bboxes = []
        for det in dets:
            bbox = det.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            bboxes.append(bbox)

        if not bboxes:
            return None, 0.0

        t_sam_start = time.perf_counter()
        input_tensor, _, orig_hw = self._preprocess_for_sam_encoder(image_bgr)

        # Use the first encoder input name instead of hardcoding.
        encoder_input_name = self.sam_encoder_session.get_inputs()[0].name
        image_embedding = self.sam_encoder_session.run(None, {encoder_input_name: input_tensor})[
            0
        ]

        boxes_xyxy = np.asarray(bboxes, dtype=np.float32)
        boxes_model = self._transform_boxes_xyxy_for_sam(boxes_xyxy, orig_hw)

        mask_logits = self._decode_sam_masks_for_boxes(
            image_embedding=image_embedding,
            boxes_xyxy_model_frame=boxes_model,
            orig_hw=orig_hw,
        )
        if mask_logits is None or mask_logits.shape[0] == 0:
            return None, 0.0

        t_sam_end = time.perf_counter()
        sam_ms = (t_sam_end - t_sam_start) * 1000.0
        return (mask_logits > self.sam_mask_threshold).astype(np.uint8), sam_ms

    def _run_yolo_onnx(self, image_bgr: np.ndarray):
        if self.onnx_session is None:
            return []

        h, w = image_bgr.shape[:2]
        image, scale_factor, pad_param = self._preprocess_for_onnx(image_bgr)

        input_ort = ort.OrtValue.ortvalue_from_numpy(
            image.transpose((0, 3, 1, 2))
        )

        if self.onnx_has_nms:
            # With NMS: [num_dets, labels, scores, boxes]
            num_dets, labels, scores, bboxes = self.onnx_session.run(
                self.onnx_output_names, {self.onnx_input_name: input_ort}
            )

            num_dets = int(num_dets[0][0])
            labels = labels[0, :num_dets]
            scores = scores[0, :num_dets]
            bboxes = bboxes[0, :num_dets]
        else:
            # Without NMS: [scores, boxes] (YOLO-World ONNX demo 스타일)
            scores_raw, bboxes_raw = self.onnx_session.run(
                self.onnx_output_names, {self.onnx_input_name: input_ort}
            )
            # scores_raw: (1, N, C), bboxes_raw: (1, N, 4)
            scores_t = torch.from_numpy(scores_raw[0])
            boxes_t = torch.from_numpy(bboxes_raw[0])

            device_torch = "cuda" if "cuda" in str(self.cfg.device) else "cpu"
            scores_t = scores_t.to(device_torch)
            boxes_t = boxes_t.to(device_torch)

            # 1) 각 박스별 최고 점수/클래스
            max_scores, labels_max = torch.max(scores_t, dim=1)
            keep_mask = max_scores > self.yolo_score_thr

            if not keep_mask.any():
                return []

            filtered_boxes = boxes_t[keep_mask]
            filtered_scores = max_scores[keep_mask]
            filtered_labels = labels_max[keep_mask]

            # 1.5) Pre-NMS Top-K 필터링으로 NMS 연산량 감소
            if len(filtered_scores) > self.yolo_topk * 2:
                _, topk_idx = torch.topk(filtered_scores, k=self.yolo_topk * 2)
                filtered_boxes = filtered_boxes[topk_idx]
                filtered_scores = filtered_scores[topk_idx]
                filtered_labels = filtered_labels[topk_idx]

            # 2) batched_nms 사용 (있으면), 없으면 단일 NMS fallback
            if batched_nms is not None:
                keep_idx = batched_nms(
                    filtered_boxes, filtered_scores, filtered_labels, iou_threshold=self.nms_iou_threshold
                )
            elif nms is not None:
                keep_idx = nms(filtered_boxes, filtered_scores, iou_threshold=self.nms_iou_threshold)
            else:
                # torchvision.ops 없음 → simple score top-k만 사용
                _, keep_idx = torch.topk(filtered_scores, k=min(self.yolo_topk, len(filtered_scores)))

            boxes_np = filtered_boxes[keep_idx].cpu().numpy()
            scores_np = filtered_scores[keep_idx].cpu().numpy()
            labels_np = filtered_labels[keep_idx].cpu().numpy()

            num_dets = int(scores_np.shape[0])
            bboxes = boxes_np
            scores = scores_np.astype(np.float32)
            labels = labels_np.astype(np.int64)

        # Undo padding / scaling
        if num_dets > 0:
            bboxes = bboxes.astype(float)
            bboxes -= np.array(
                [pad_param[1], pad_param[0], pad_param[1], pad_param[0]]
            )
            bboxes /= scale_factor
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
            bboxes = bboxes.round().astype("int")

        dets = []
        for i in range(num_dets):
            x1, y1, x2, y2 = bboxes[i].tolist()
            class_id = int(labels[i])
            score = float(scores[i])

            name = str(class_id)
            try:
                if 0 <= class_id < len(self.yolo_texts):
                    name = self.yolo_texts[class_id][0]
            except Exception:
                pass

            if self.allowed_class_ids:
                if class_id not in self.allowed_class_ids:
                    continue
            elif self.allowed_object_labels:
                if str(name).strip().lower() not in self.allowed_object_labels:
                    continue

            dets.append(
                {
                    "label": name,
                    "score": score,
                    "bbox": [x1, y1, x2, y2],
                    "class_id": class_id,
                }
            )

        return dets


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloWorldOnnxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()