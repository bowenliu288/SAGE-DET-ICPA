# -*- coding: utf-8 -*-
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ROOT = r"E:\yolo26_CBDDownLite_P5SemanticGate_head"
sys.path.insert(0, ROOT)

from ultralytics import YOLO

if __name__ == "__main__":
    yaml_path = r"E:\yolo26_CBDDownLite_P5SemanticGate_head\ultralytics\cfg\models\26\yolo26_v7_pgq_lite_head.yaml"
    print("=" * 80)
    print("[Check] Building PGQ-Lite Head model from:")
    print(yaml_path)
    print("=" * 80)
    model = YOLO(yaml_path)
    print("\n[OK] Model was built successfully.")
    print("[Check] Last layer:")
    print(model.model.model[-1])
    print("\n[OK] PGLiteDetect check finished.")
