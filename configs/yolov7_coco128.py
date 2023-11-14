import os
# ----------------------------------------
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
# --------------------------------------------------


YOLO_DEVICE = "cuda:0"
# YOLO_DEVICE = "cpu"
YOLO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
]
YOLO_NET_CONF = os.path.join(CURRENT_DIR, 'yolov7.yaml')
YOLO_WEIGHT_PATH = 'weights/yolov7.pkl'
YOLO_TARGET_SIZE = (640, 640)
YOLO_PADDING_COLOR = (114, 114, 114)
YOLO_THRESHOLD_CONF = 0.25
YOLO_THRESHOLD_IOU = 0.45
YOLO_MULTI_LABLE = False
SCALE_FILL = False       # False是前处理时对图像keep_ratio方式resize
BATCH_FILL = True
PICK_CLASSES = [0, 2, 5, 7]       # 挑选类别的序号