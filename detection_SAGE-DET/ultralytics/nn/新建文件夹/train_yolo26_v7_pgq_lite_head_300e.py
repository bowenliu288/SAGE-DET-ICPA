# -*- coding: utf-8 -*-
import warnings
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, r"E:\yolo26_CBDDownLite_P5SemanticGate_head")
warnings.filterwarnings("ignore")

from ultralytics import YOLO

if __name__ == "__main__":
    yaml_path = r"E:\yolo26_CBDDownLite_P5SemanticGate_head\ultralytics\cfg\models\26\yolo26_v7_pgq_lite_head.yaml"
    model = YOLO(yaml_path)
    model.train(
        data=r"E:\yolo26_CBDDownLite_P5SemanticGate_head\train\train_drone_selfdataset.yaml",
        cache=False,
        imgsz=640,
        epochs=300,
        batch=16,
        workers=4,
        close_mosaic=0,
        device=os.environ.get("CUDA_VISIBLE_DEVICES", 0),
        optimizer="MuSGD",
        patience=0,
        amp=True,
        cos_lr=False,
        save_period=-1,
        project=r"E:\yolo26_CBDDownLite_P5SemanticGate_head\runs\train_drone_selfdataset",
        name="yolo26_v7_pgq_lite_head_300e",
    )
