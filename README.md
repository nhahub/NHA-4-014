# NHA-4-014 - Autonomous Car Object Detection

This repository contains a complete computer vision graduation project for road-object detection with YOLOv8. The project starts with dataset exploration, prepares YOLO-format training data, trains and evaluates multiple YOLO models, tracks experiments with MLflow, and packages a Windows WPF desktop detector that can run on a local camera or MP4 video.

## Project Goal

The goal is to detect autonomous-driving road objects in images and video using a YOLOv8 object detection model. The project focuses on 10 vehicle and road-user classes:

1. Bus
2. Cyclist
3. Light Vehicle
4. Long Truck
5. Medium Truck
6. Minibus Taxi
7. Motorcycle
8. Pedestrian
9. Person with wheel barrow
10. Short Truck

The dataset is handled in YOLO format, with image folders and matching label files using normalized bounding boxes.

## Repository Structure

```text
.
|-- 1-Data Exploration/
|   `-- EDA.ipynb
|-- 2-Model Training/
|   |-- Data Prepration.ipynb
|   |-- Yolo Model Training.ipynb
|   `-- models/
|       |-- model_v1.pt
|       `-- model_v2.pt
|-- 3-Deployment/
|   |-- YoloWpfDetector.rar
|   `-- net8.0-windows.rar
|-- 4-ML Flow/
|   |-- autonomous_car_detection.ipynb
|   `-- MLflow_Experiment_Report.pdf
`-- README.md
```

## 1. Data Exploration

Notebook: `1-Data Exploration/EDA.ipynb`

This notebook explores the YOLO dataset before training. It mounts Google Drive in Colab, reads image and label folders, loads `data.yaml`, and analyzes the training annotations.

Main analysis steps:

- Reads YOLO label files from `train/labels`.
- Maps numeric class IDs to class names.
- Builds a Pandas dataframe of bounding boxes.
- Plots class distribution.
- Plots bounding box area distribution.
- Counts objects per image.
- Visualizes sample images with bounding boxes using OpenCV.

The purpose of this stage is to understand class balance, object sizes, and whether small objects such as pedestrians, cyclists, and motorcycles may need special attention during training.

## 2. Data Preparation

Notebook: `2-Model Training/Data Prepration.ipynb`

This notebook prepares the dataset for YOLO training. It uses Google Colab and Google Drive paths, then creates a YOLO-compatible dataset directory.

Main preparation steps:

- Loads the original train images, train labels, and `data.yaml`.
- Reads class names and annotation statistics.
- Performs exploratory plots similar to the EDA notebook:
  - class distribution
  - bounding box area distribution
  - objects per image
  - aspect ratio distribution
  - object position heatmap
- Splits the dataset into `train` and `val` folders.
- Writes a generated `dataset/data.yaml`.
- Trains early YOLOv8n experiments.

The generated YOLO config uses this structure:

```yaml
path: dataset
train: train/images
val: val/images
nc: 10
names:
  - Bus
  - Cyclist
  - Light Vehicle
  - Long Truck
  - Medium Truck
  - Minibus Taxi
  - Motorcycle
  - Pedestrian
  - Person with wheel barrow
  - Short Truck
```

Training experiments in this notebook include YOLOv8n runs with:

- `epochs=5`, `imgsz=640`, `batch=16`
- `epochs=20`, `imgsz=640`, `batch=16`

## 3. Model Training

Notebook: `2-Model Training/Yolo Model Training.ipynb`

This notebook contains local Windows training and inference workflows using Ultralytics YOLOv8.

Main workflow:

- Checks Python and PyTorch environment.
- Uses `ultralytics.YOLO`.
- Builds `data.yaml` for a local dataset path.
- Trains YOLOv8n on the vehicle dataset.
- Saves trained weights.
- Runs prediction on validation images.
- Runs prediction on a test video.

Important training configuration:

```python
model = YOLO("yolov8n.pt")
model.train(
    data=r"D:\YOLO Model\vehicle dataset\dataset\data.yaml",
    epochs=5,
    imgsz=640,
    batch=8,
    device=0,
    name="local_run"
)
```

A later run trains for 20 epochs at `imgsz=640` with `batch=8`.

The notebook output shows YOLOv8n training on an NVIDIA GeForce GTX 1660 Ti. The 5-epoch local run reached approximately:

| Metric | Value |
| --- | ---: |
| Precision | 0.7089 |
| Recall | 0.6318 |
| mAP50 | 0.6638 |
| mAP50-95 | 0.5070 |

## Trained Models

Folder: `2-Model Training/models/`

The repository includes two PyTorch YOLO model weight files:

| File | Size | Purpose |
| --- | ---: | --- |
| `model_v1.pt` | 6.2 MB | First exported trained model |
| `model_v2.pt` | 6.2 MB | Later exported trained model |

These `.pt` files are binary model weights. They are used by the deployment app and can also be loaded directly with Ultralytics:

```python
from ultralytics import YOLO

model = YOLO("2-Model Training/models/model_v2.pt")
results = model.predict("path/to/image_or_video.mp4", conf=0.25, imgsz=640)
```

## 4. MLflow Experiment Tracking

Notebook: `4-ML Flow/autonomous_car_detection.ipynb`

Report: `4-ML Flow/MLflow_Experiment_Report.pdf`

This notebook turns the training process into tracked MLflow experiments. It installs `ultralytics`, `mlflow`, and `pyngrok`, trains multiple YOLOv8 variants, logs parameters and metrics, compares runs, and exposes the MLflow UI through ngrok when running in Colab.

The notebook compares three main experiments:

| Run | Model | Epochs | Image Size | Batch | Main Change |
| --- | --- | ---: | ---: | ---: | --- |
| Run 1 - Baseline | YOLOv8n | 40 | 640 | 16 | Baseline model |
| Run 2 - imgsz=800 | YOLOv8n | 50 | 800 | 16 | Larger input resolution |
| Run 3 - YOLOv8s | YOLOv8s | 50 | 640 | 8 | Larger model |

Final MLflow comparison:

| Metric | Run 1 Baseline | Run 2 imgsz=800 | Run 3 YOLOv8s | Best |
| --- | ---: | ---: | ---: | --- |
| mAP50 | 0.4676 | 0.7912 | 0.7922 | Run 3 |
| mAP50-95 | 0.3393 | 0.6078 | 0.5991 | Run 2 |
| Precision | 0.6613 | 0.8382 | 0.8201 | Run 2 |
| Recall | 0.4517 | 0.6860 | 0.7287 | Run 3 |

The report concludes that Run 3 is the best overall autonomous-driving model because it has the highest recall. In this domain, missing an object is usually more dangerous than producing an occasional false detection. Run 2 remains important because it gives the best precision and box localization score.

The report also notes that motorcycles remain challenging, likely because of class imbalance, and recommends collecting more motorcycle examples or augmenting minority classes.

## 5. Deployment

Folder: `3-Deployment/`

The deployment folder contains two RAR archives:

| Archive | Contents |
| --- | --- |
| `YoloWpfDetector.rar` | Full WPF project source, Python worker, requirements, models, build outputs, and solution files |
| `net8.0-windows.rar` | Built Windows output folder ready to run |

The WPF app is named **YOLO Road Object Detector**. It is built with:

- .NET 8
- WPF
- Python worker process
- Ultralytics YOLO
- OpenCV

Main app features:

- Select local camera or MP4 video input.
- Select an available `.pt` model from the `models` folder.
- Run without a model as a camera/video preview.
- Control confidence threshold from 5% to 95%.
- Limit FPS or run at maximum available FPS.
- Display model classes.
- Check and install Python requirements from the UI.
- Stream annotated frames from Python to WPF.

The deployment Python requirements are:

```text
ultralytics
opencv-python
```

### How the Desktop App Works

The WPF app starts `detector_worker.py` as a child Python process. The worker:

1. Opens the camera or video with OpenCV.
2. Loads the selected YOLO `.pt` model if one is chosen.
3. Runs `model.predict(...)` on each frame.
4. Draws detections on the frame.
5. Encodes the frame as JPEG.
6. Streams the JPEG bytes back to WPF through standard output.

The C# app reads each frame, converts it into a `BitmapImage`, and displays it in the UI.

## How to Run the Notebooks

Most notebooks were written for Google Colab and expect the dataset in Google Drive.

Recommended Colab setup:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Install dependencies:

```python
!pip install -q ultralytics mlflow pyngrok yt-dlp
```

Expected dataset layout:

```text
vehicle dataset/
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/ or val/
|   |-- images/
|   `-- labels/
`-- data.yaml
```

If running locally, update hardcoded paths such as:

```python
BASE_PATH = "/content/drive/MyDrive/vehicle dataset"
DATA_PATH = r"D:\YOLO Model\vehicle dataset"
```

## How to Run the Windows Detector

1. Extract `3-Deployment/net8.0-windows.rar`.
2. Open the extracted `net8.0-windows` folder.
3. Make sure Python is installed and available as `python`.
4. Run `YoloWpfDetector.exe`.
5. Click **Check / Install Requirements** if packages are missing.
6. Select:
   - local camera or MP4 video
   - YOLO model
   - confidence threshold
   - FPS limit
7. Click **Start**.

The app expects this kind of output folder:

```text
net8.0-windows/
|-- YoloWpfDetector.exe
|-- detector_worker.py
|-- requirements.txt
`-- models/
    |-- model_v1.pt
    `-- model_v2.pt
```

## Development Notes

- The notebooks contain both Colab and local Windows workflows.
- Several paths are machine-specific and should be changed before rerunning.
- The deployment source is currently stored inside `YoloWpfDetector.rar`, not as an extracted project folder.
- The `.pt` model files are binary weights and cannot be reviewed like source code.
- The built deployment archive includes generated `bin`, `obj`, and `.vs` files.

## Main Results

The strongest tracked model is Run 3, using YOLOv8s:

- mAP50: `0.7922`
- mAP50-95: `0.5991`
- Precision: `0.8201`
- Recall: `0.7287`

Run 2, using YOLOv8n with `imgsz=800`, is also valuable:

- Best precision: `0.8382`
- Best mAP50-95: `0.6078`

Overall, increasing image size gave the largest improvement over the baseline, while using YOLOv8s gave the best recall and was selected as the best overall model for autonomous-driving safety.
