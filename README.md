# 합성데이터 기반 객체 탐지 AI 경진대회
## 결과
* **★ RIVATE 62위 top 8%★** 
* base model : Yolov8x 

## 1. Introduction

**[주제]** 합성 데이터를 활용한 자동차 탐지 AI 모델 개발

**[기간]** 2023.05.08 ~ 2023.06.19 09:59 

**[링크]** [https://dacon.io/competitions/official/236118/overview/description](https://dacon.io/competitions/official/236107/overview/description)
## 2. Data
```
data
├─  train
│   ├─  img : 6481개
│   │   ├─  syn_000000.png
│   │   ├─  syn_000001.png
│   │   └─  ...
│   └─  annotation  : 6481개 ( class_id, LabelMe 형식의 Bounding Box 좌표)
│       ├─  syn_00000.txt
│       ├─  syn_00001.txt
│       └─  ...
├─  test
|   ├─  img : 3400개
|   │   ├─  syn_000000.png 
|   │   ├─  syn_000001.png 
|   └─  └─  ...
| 
├─  class.txt
│   ├─  class_id 
|   └─  class_name

├─  sample_submission.csv 
|   ├─ class_id : 검출한 객체 id
|   ├─confidence : 검출한 객체의 정확도(0~1)
|   ├─point1_x : 검출한 객체 좌상단 x좌표
|   ├─point1_y : 검출한 객체 좌상단 y좌표
|   ├─point2_x : 검출한 객체 우상단 x좌표
|   ├─point2_y : 검출한 객체 우상단 y좌표
|   ├─point3_x : 검출한 객체 우하단 x좌표
|   ├─point3_y : 검출한 객체 우하단 y좌표
|   ├─point4_x : 검출한 객체 좌하단 x좌표
└─  └─point4_y : 검출한 객체 좌하단 y좌표
```

## 2-1. Labelme to Yolo
```python
def make_yolo_dataset(image_paths, txt_paths, type="train"):
    for image_path, txt_path in tqdm(zip(image_paths, txt_paths if not type == "test" else image_paths), total=len(image_paths)):
        source_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_height, image_width, _ = source_image.shape

        target_image_path = f"input/{type}/{os.path.basename(image_path)}"
        cv2.imwrite(target_image_path, source_image)

        if type == "test":
            continue

        with open(txt_path, "r") as reader:
            yolo_labels = []
            for line in reader.readlines():
                line = list(map(float, line.strip().split(" ")))
                class_name = int(line[0])
                x_min, y_min = float(min(line[5], line[7])), float(min(line[6], line[8]))
                x_max, y_max = float(max(line[1], line[3])), float(max(line[2], line[4]))
                x, y = float(((x_min + x_max) / 2) / image_width), float(((y_min + y_max) / 2) / image_height)
                w, h = abs(x_max - x_min) / image_width, abs(y_max - y_min) / image_height
                yolo_labels.append(f"{class_name} {x} {y} {w} {h}")

        target_label_txt = f"input/{type}/{os.path.basename(txt_path)}"
        with open(target_label_txt, "w") as writer:
            for yolo_label in yolo_labels:
                writer.write(f"{yolo_label}\n")

```
## 2-2. Yolo to Labelme 
```python
def yolo_to_labelme(line, image_width, image_height, txt_file_name):
    file_name = txt_file_name.split("/")[-1].replace(".txt", ".png")
    class_id, x, y, width, height, confidence = [float(temp) for temp in line.split()]

    x_min = int((x - width / 2) * image_width)
    x_max = int((x + width / 2) * image_width)
    y_min = int((y - height / 2) * image_height)
    y_max = int((y + height / 2) * image_height)

    return file_name, int(class_id), confidence, x_min, y_max, x_max, y_max, x_max, y_min, x_min, y_min
```
## 3. Setup
* In Colab-PRO 
* Set up for  GPU T4

 ### Clone Yolo
```python
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
```

### Install
```python
%pip install -qr requirements.txt
```


### Make yaml file
```python
import yaml

yaml_data = {'nc': len(classes),
             'names': classes,
             "path": root_dir,
             "train":os.path.join(root_dir, "train.txt"), # train 경로
             "val":os.path.join(root_dir, "valid.txt") # valid 경로
             }

with open(os.path.join(root_dir, "custom.yaml"), "w") as f:
  yaml.dump(yaml_data, f)
```
## 4. Run
**Freeze backbone**
![image](https://github.com/Jellyjellyjinjin/Daycon-car-object-detection-/assets/118363210/10f5055f-3f2c-428a-8a51-c9379eb407ff)


```python
%%time
%cd /content/yolov5

if TRAIN:
    !python train.py --data '/content/input/custom.yaml' --weights yolov5x.pt \
    --img 640 --epochs {EPOCHS} --batch-size 8 --name 'yolov5x_result' \
    --freeze 10 --project=f"{savepath}/freeze" \
    --optimizer Adam --seed 42 --hyp /content/yolov5/data/hyps/hyp.scratch-low.yaml
```
**hyperparameters**
 * 학습 : lr0=0.001, lrf=0.01, warmup_epochs=3.0
 * 색 : hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
 * 이미지: mosaic=1.0, degrees=0.0(image rotation), translate=0.0, scale=0.0, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.0, mixup=0.0, 

## 5. Inference
```python
!python detect.py --weights /content/drive/MyDrive/ATL/weight/freeze/best.pt --img 640  --source /content/input/test --save-conf --save-txt
```

## 6. submit
```python
import glob
import cv2
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

submit = pd.DataFrame(columns=['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y', 'point3_x', 'point3_y', 'point4_x', 'point4_y'])

for txt in tqdm(infer_txt_list):
    with open(txt, 'r') as f:
        lines = f.readlines()
        base_file_name = txt.split('/')[-1].split('.')[0]
        img_height, img_width = cv2.imread('/content/yolov5/runs/detect/exp/' + base_file_name + '.png').shape[:2]
        for line in lines:
            file_name, class_id, confidence, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y = yolo_to_labelme(line, img_width, img_height, txt)
            submit = submit.append({'file_name':file_name, 'class_id':class_id, 'confidence':confidence, 'point1_x':point1_x, 'point1_y':point1_y, 'point2_x':point2_x, 'point2_y':point2_y, 'point3_x':point3_x, 'point3_y':point3_y, 'point4_x':point4_x, 'point4_y':point4_y}, ignore_index=True)
```
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Data_mixAug
**Set transform**
```python
train_tfms = A.Compose([
        A.OneOf([
            A.ISONoise(p=1.0, intensity=(0.39, 0.82)),
            A.GaussNoise(p=1.0, var_limit=(15.0, 50.0))
        ], p=0.5),

        A.OneOf([
            A.MotionBlur(p=1.0, blur_limit=(5, 15)),
            A.GaussianBlur (p=1.0, blur_limit = (3,9)),
            A.ImageCompression(p=1.0, quality_lower=25, quality_upper=40),
        ], p=0.4),


        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.6)
      ])
  ```

**Apply to image**
```python
def apply_transforms_and_save(input_file_path, output_folder):
  image = cv2.imread(input_file_path)
  transformed_image = train_tfms(image=image)["image"]

    # NumPy 배열로 변환
  transformed_image = np.array(transformed_image)
  output_file_path = os.path.join(output_folder, input_file_path[-13:])
  cv2.imwrite(output_file_path, transformed_image)
```
