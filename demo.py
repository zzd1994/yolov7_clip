from algorithms.yolov7 import YOLOv7Detector
import configs.yolov7_coco128 as yolov7_config
import os, cv2, random, time


# detect model
detector = YOLOv7Detector.from_config(yolov7_config)
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yolov7_config.YOLO_CLASSES))]
colors_dict = dict(zip(yolov7_config.YOLO_CLASSES, colors))

img_path = r"D:\datasets\highway_20230816\pick_temp"
# img_path = r'D:\datasets\highway_20230816\camera2_images_pick_crop_part'
img_list = sorted(os.listdir(img_path))

for img_name in img_list[:]:
    img = cv2.imread(os.path.join(img_path, img_name))
    img_copy = img.copy()
    pred = detector.det([img])[0]
    for klass_name, score, (x1, y1, x2, y2), (xo, yo, w, h) in pred:
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), colors_dict[klass_name], 3)
        score = round(score, 3)
        cv2.putText(img_copy, f'{klass_name}: {score}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.namedWindow('frame', 0)
    cv2.imshow('frame', img_copy)
    cv2.waitKey(0)
