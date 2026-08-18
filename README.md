# ☁️ Cloud Image Analysis and Classification

A Python-based image processing project for **cloud detection, feature extraction, and cloud-type classification** using computer vision and image processing techniques.

The project analyzes sky/cloud images and extracts characteristics such as cloud density, texture, edge density, color information, and brightness variation. These features can later be used for statistical analysis or machine-learning-based cloud classification.

---

## 📌 Project Overview

The project consists of two main stages:

### Stage 1 — Basic Cloud Classification

The original implementation processes an individual cloud image by:

1. Loading the image.
2. Converting it to grayscale.
3. Removing extremely bright pixels caused by sunlight or other illumination.
4. Applying Otsu's thresholding.
5. Calculating cloud density.
6. Classifying the image according to predefined density ranges.
7. Displaying the original image, filtered grayscale image, and threshold mask.

The classification is based on cloud density.

### Stage 2 — Feature Extraction

The enhanced implementation processes multiple images and extracts several numerical features:

* Cloud density
* GLCM contrast
* GLCM homogeneity
* GLCM energy
* GLCM correlation
* Edge density
* Blue-channel ratio
* Brightness standard deviation

The extracted features and automatically generated labels are saved into `features.csv`.

---

## 🌥️ Cloud Classes

The current classification uses cloud density thresholds:

| Cloud Density | Classification |
| ------------: | -------------- |
|        `< 5%` | Clear          |
|   `5% – <20%` | Cirrus         |
|  `20% – <40%` | Cumulus        |
|  `40% – <60%` | Stratus        |
|        `≥60%` | Overcast       |

This classification logic is implemented in both the original classifier and the automatic labelling function.

> **Note:** These labels are generated using cloud-density thresholds. They should not be considered ground-truth meteorological classifications without validation against manually labelled data.

---

## 🔬 Features Extracted

### 1. Cloud Density

Cloud density represents the percentage of image pixels considered brighter than the calculated Otsu threshold.

```text
Cloud Density = Bright Pixels / Total Pixels × 100
```

Higher density generally indicates greater cloud coverage.

---

### 2. GLCM Texture Features

The project uses a **Gray-Level Co-occurrence Matrix (GLCM)** to describe cloud texture.

The grayscale image is reduced from 256 gray levels to 64 levels before constructing the GLCM.

The following features are extracted:

#### Contrast

Measures differences between neighboring pixels.

* High contrast → more varied/jagged texture
* Low contrast → smoother texture

#### Homogeneity

Measures how uniform the neighboring pixel values are.

* High homogeneity → smoother and more uniform regions

#### Energy

Measures the regularity or uniformity of the texture.

#### Correlation

Measures the linear relationship between neighboring pixels.

---

### 3. Edge Density

Canny edge detection is used to identify edges in the image.

```text
Edge Density = Number of Edge Pixels / Total Pixels × 100
```

Higher edge density indicates a larger amount of visible structure or variation in the cloud scene.

---

### 4. Blue Ratio

The blue-channel ratio is calculated from the BGR image:

```text
Blue Ratio = B / (B + G + R)
```

A relatively high blue ratio can indicate clearer sky, while cloud regions tend to have more similar RGB channel values.

---

### 5. Brightness Standard Deviation

The standard deviation of grayscale brightness is calculated to represent variation in illumination across the image.

Higher values indicate greater brightness variation.

---

## 🧹 Image Preprocessing

Before feature extraction, the image undergoes several preprocessing steps.

### Grayscale Conversion

The input BGR image is converted to grayscale:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

### Bright-Pixel Filtering

The top approximately 2% brightest pixels are identified using the 98th percentile and replaced with the median grayscale value.

This is intended to reduce the influence of extremely bright regions such as sunlight or lightning.

### Otsu Thresholding

Otsu's method automatically determines a threshold for separating brighter and darker regions of the image.

The resulting threshold is then used to calculate cloud density.

---

## 📁 Project Structure

```text
Cloud-Image-Analysis/
│
├── images/
│   ├── pic_0.jpg
│   ├── pic_1.jpg
│   ├── pic_2.jpg
│   ├── ...
│   └── pic_25.jpg
│
├── Try05.py
├── stage1_features.py
├── features.csv
├── feature_overview.png
└── README.md
```

### Files

| File                   | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `Try05.py`             | Original single-image cloud classification implementation |
| `stage1_features.py`   | Multi-image feature extraction pipeline                   |
| `images/`              | Input cloud/sky images                                    |
| `features.csv`         | Extracted numerical features and automatic labels         |
| `feature_overview.png` | Visualization of extracted features                       |
| `README.md`            | Project documentation                                     |

---

## ⚙️ Requirements

Python 3.x is required.

Install the required libraries using:

```bash
pip install opencv-python numpy matplotlib scikit-image
```

The project uses:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
```

The feature-extraction script also uses Python's built-in:

```python
import csv
import os
```

---

## 🚀 How to Run

### 1. Prepare the Images

Create an `images` directory in the project root:

```text
Cloud-Image-Analysis/
└── images/
```

Place the input images inside it using the expected naming format:

```text
pic_0.jpg
pic_1.jpg
pic_2.jpg
...
pic_25.jpg
```

The feature-extraction script checks for these filenames and skips images that do not exist.

---

### 2. Run the Original Classifier

To process the example image used by `Try05.py`:

```bash
python Try05.py
```

The program displays:

1. Original image
2. Filtered grayscale image
3. Otsu threshold mask

It also prints:

```text
Otsu's Optimal Threshold
Cloud Density
Classified Cloud Type
```

The original implementation uses `images/pic_11.jpg`.

---

### 3. Run Feature Extraction

Run:

```bash
python stage1_features.py
```

The program processes the available images and extracts the feature set.

The output is saved as:

```text
features.csv
```

The script also generates:

```text
feature_overview.png
```

The CSV-writing process is implemented in the feature extraction pipeline.

---

## 📊 Example CSV Structure

The generated `features.csv` contains columns similar to:

```text
image
density
contrast
homogeneity
energy
correlation
edge_density
blue_ratio
brightness_std
label
```

Example:

```text
pic_1.jpg,23.4512,1.8234,0.7412,0.5123,0.8421,4.2312,0.4211,38.2213,Cumulus
```

---

## 📈 Feature Visualization

After processing the images, the program creates a feature overview chart.

The visualization includes:

* Density
* Contrast
* Homogeneity
* Edge Density
* Blue Ratio

Each image is represented on the x-axis, allowing feature values to be compared between images.

Output:

```text
feature_overview.png
```

---

## 🔄 Processing Pipeline

```text
              Input Cloud Image
                     │
                     ▼
              Image Loading
                     │
                     ▼
             Grayscale Conversion
                     │
                     ▼
          Bright-Pixel Suppression
                     │
                     ▼
             Otsu Thresholding
                     │
          ┌──────────┴───────────┐
          ▼                      ▼
   Cloud Density           Image Features
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
             GLCM        Edge Density    Blue Ratio
           Features                       │
               │                          ▼
               └──────────┬───────────────┘
                          ▼
                Brightness Statistics
                          │
                          ▼
                  Automatic Label
                          │
                          ▼
                    features.csv
                          │
                          ▼
                 Feature Visualization
```

---

## 🧠 Current Methodology

The current project is primarily an **image processing and feature engineering system**, rather than a machine-learning classifier.

The automatically generated label is based only on cloud density:

```python
label = auto_label(density)
```

Therefore, the extracted features can be viewed as a foundation for a future machine-learning stage.

---

## 🔮 Future Improvements

Possible future extensions include:

### Machine Learning

Use the extracted features to train classifiers such as:

* Decision Tree
* Random Forest
* Support Vector Machine
* K-Nearest Neighbors
* Logistic Regression
* Neural Network

### Better Ground-Truth Labels

Instead of generating labels from density thresholds, manually label the images or use a validated cloud dataset.

### Additional Features

Potential features include:

* Local Binary Patterns (LBP)
* Histogram of Oriented Gradients (HOG)
* RGB/HSV color statistics
* Cloud shape descriptors
* Histogram features
* Entropy
* Morphological features

### Model Evaluation

Evaluate the classification system using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

---

## ⚠️ Limitations

1. **Density-based labels are heuristic labels**, not validated meteorological ground truth.
2. Lighting conditions can influence grayscale intensity and cloud density.
3. The blue-ratio feature can be affected by camera characteristics and atmospheric conditions.
4. GLCM features depend on image texture and preprocessing.
5. The current implementation does not train a machine-learning model.
6. The current image-processing pipeline assumes the input images follow the expected naming convention.

---

## 👨‍💻 Technologies Used

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**
* **scikit-image**
* **GLCM**
* **Otsu Thresholding**
* **Canny Edge Detection**
* **CSV Data Processing**

---

## 🎯 Project Goal

The primary goal of this project is to transform cloud images into a structured set of numerical features that can be used to analyze and classify different sky/cloud conditions.

The feature-extraction stage provides the foundation for a subsequent **machine-learning-based cloud classification system**.


## Author

Achala Dasanayake — Faculty of Science, University of Peradeniya
