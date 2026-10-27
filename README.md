# UTKFace Facial Attribute Analysis with PyTorch

Convolutional Neural Network (CNN) implementations in PyTorch for facial attribute classification and regression on the **UTKFace** dataset (`23,705` aligned facial images).

The project is organized into two stages:
1. **Exploratory Data Analysis (`UTKFace_1.ipynb`)**: Data integrity verification, target distribution profiling, outlier detection, class imbalance analysis, and channel-wise RGB statistics.
2. **Model Training & Evaluation (`UTKFace_2.ipynb`)**:
   - **`BinaryGenderCNN`**: Binary gender classification (`Male` vs. `Female`).
   - **`MultiClassRaceCNN`**: 5-class ethnicity classification (`White`, `Black`, `Asian`, `Indian`, `Others`) with inverse-frequency class weights.
   - **`AgeRegressionCNN`**: Continuous age estimation (`1–116` years) using Huber Loss (`SmoothL1Loss`).
   - **`MultiTaskFaceResNet`**: Joint multi-task learning on a shared `ResNet-18` backbone predicting gender, race, and age simultaneously.

---

## Repository Structure

```text
utkface/
├── UTKFace_1.ipynb          # Stage 1: Exploratory Data Analysis (EDA)
├── UTKFace_2.ipynb          # Stage 2: Custom CNNs & Multi-Task ResNet-18 Training
├── REPORT.md                # Full technical report and evaluation analysis
├── README.md                # Project overview and experimental results
└── assets/                  # Figures generated from the notebooks
    ├── 01_eda_age_distribution.png
    ├── 02_eda_gender_race_distribution.png
    ├── 03_eda_multivariate_age_gender_race.png
    ├── 04_eda_sample_faces.png
    ├── 05_cnn_binary_gender_results.png
    ├── 06_cnn_multiclass_race_results.png
    ├── 07_cnn_age_regression_results.png
    ├── 08_multitask_resnet18_test_predictions.png
    └── 09_custom_image_inference.png
```

---

## Stage 1: Exploratory Data Analysis (`UTKFace_1.ipynb`)

### 1. Data Cleaning

The `jangedoo/utkface-new` Kaggle archive contains duplicate directories (`UTKFace/`, `utkface_aligned_cropped/UTKFace/`, and `crop_part1/`). To prevent data leakage across splits, only the primary `UTKFace/` directory (`23,708` images) is indexed.

- **Filename label format**: `[age]_[gender]_[race]_[date_time].jpg`
  - `age`: Integer in `[1, 116]`
  - `gender`: `0` = Male, `1` = Female
  - `race`: `0` = White, `1` = Black, `2` = Asian, `3` = Indian, `4` = Others

Three malformed files missing the `race` field were excluded (`61_1_20170109142408075.jpg.chip.jpg`, `61_1_20170109150557335.jpg.chip.jpg`, `39_1_20170116174525125.jpg.chip.jpg`), leaving **`23,705` valid images**.

### 2. Age Distribution (Regression Target)

| Metric | Value | Notes |
| :--- | :---: | :--- |
| **Sample Size ($N$)** | `23,705` | Cleaned dataset |
| **Range (Min – Max)** | `1 – 116` | Strictly positive continuous target |
| **Mean ± Std** | `33.30 ± 19.89` | Wide age dispersion |
| **Quartiles ($Q_1$ / Median / $Q_3$)** | `23.0` / `29.0` / `45.0` | Interquartile range $\text{IQR} = 22.0$ years |
| **Skewness** | `0.622` | Right-skewed distribution |
| **Outlier Threshold ($Q_3 + 1.5 \times \text{IQR}$)** | `> 78.0` years | **`696` images (`2.94%`)** in the `79–116` age bracket |

![Age Distribution](assets/01_eda_age_distribution.png)

### 3. Gender & Race Distributions (Classification Targets)

Class imbalance is measured using the Imbalance Ratio (`IR = max(N_c) / min(N_c)`):

- **Gender (`IR = 1.10x`)**: `Male` = `12,391` (`52.27%`), `Female` = `11,314` (`47.73%`). The distribution is balanced.
- **Race (`IR = 5.96x`)**: Imbalanced toward `White` (`42.51%`) vs. `Others` (`7.14%`). Balanced class weights are computed as `w_c = N_total / (C * N_c)`:

| Class ID | Ethnicity | Count ($N_c$) | Percentage | Class Weight ($w_c$) |
| :---: | :--- | :---: | :---: | :---: |
| `0` | **White** | `10,078` | `42.51%` | `0.470` |
| `1` | **Black** | `4,526` | `19.09%` | `1.048` |
| `2` | **Asian** | `3,434` | `14.49%` | `1.381` |
| `3` | **Indian** | `3,975` | `16.77%` | `1.193` |
| `4` | **Others** | `1,692` | `7.14%` | `2.802` |

![Gender and Race Distributions](assets/02_eda_gender_race_distribution.png)

### 4. Multivariate Analysis & Stratified Splitting

Age distributions vary across demographic groups (`Male` median age is `34.0` vs. `26.0` for `Female`; `White` median age is `35.0` vs. `25.0` for `Others`). The dataset is partitioned into **70% Train (`16,593`), 15% Validation (`3,556`), and 15% Test (`3,556`)** stratified on the composite `gender_race` key.

![Age by Gender and Race](assets/03_eda_multivariate_age_gender_race.png)

### 5. Image Properties & RGB Statistics

Sampling `1,000` random images confirms uniform `200×200` RGB dimensions:
- **Channel Mean ($\mu_{RGB}$)**: `[0.5926, 0.4526, 0.3864]`
- **Channel Std ($\sigma_{RGB}$)**: `[0.2573, 0.2288, 0.2246]`

![Sample Faces](assets/04_eda_sample_faces.png)

---

## Architecture & Training Setup

| Component | Configuration | Rationale |
| :--- | :--- | :--- |
| **Input Pipeline** | `Resize(128, 128)`, `Normalize(UTK_MEAN, UTK_STD)` | Preserves `1:1` aspect ratio while reducing compute cost; uses dataset-specific RGB statistics. |
| **Custom CNN Backbone** | 4 `ConvBlock` layers (`3 → 32 → 64 → 128 → 256`) + `AdaptiveAvgPool2d(1)` + `FC(256 → 128)` (`~422K` params) | Progressive spatial downsampling (`128 → 64 → 32 → 16 → 8`) with global average pooling to limit parameter count. |
| **Hidden Activation & Init** | `LeakyReLU(0.01)` + Kaiming He Normal | Prevents dead neurons from large gradients and stabilizes variance across convolutional layers. |
| **Task 1: Gender** | `Linear(128 → 1)` + `Sigmoid`<br/>`nn.BCEWithLogitsLoss()` | Unweighted binary cross-entropy for balanced binary targets. |
| **Task 2: Race** | `Linear(128 → 5)` + `Softmax`<br/>`nn.CrossEntropyLoss(weight=class_weights)` | Penalizes misclassification on minority classes (`Others`, `Asian`, `Indian`). |
| **Task 3: Age** | `Linear(128 → 1)` (Identity)<br/>`nn.SmoothL1Loss(beta=5.0)` | Quadratic for small errors (under `5` years) and linear for large errors to reduce sensitivity to elderly age outliers. |
| **Task 4: Multi-Task ResNet-18** | Pretrained `ResNet-18` backbone (`11.37M` params) + 3 task heads | Combines transfer learning and shared feature representation; scales age by `1/100` to balance task gradients. |

---

## Stage 2: Experimental Results (`UTKFace_2.ipynb`)

All models were trained on Google Colab (NVIDIA Tesla T4 GPU) using `AdamW` (`weight_decay = 1e-4`) and `ReduceLROnPlateau`.

### 1. Test Set Comparison (`3,556` Images)

| Model | Gender Acc (%) | Gender Macro F1 (%) | Race Acc (%) | Race Macro F1 (%) | Age MAE (years) | Age RMSE (years) | Age $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Custom CNN (3 Separate Models)** | `89.57` | `89.53` | `72.13` | `65.01` | `7.53` | `10.46` | `0.7261` |
| **2. Multi-Task ResNet-18 (Joint Model)** | **`92.15`** | **`92.15`** | **`78.26`** | **`73.52`** | **`5.49`** | **`7.67`** | **`0.8531`** |
| **Improvement** | **`+2.58`** | **`+2.62`** | **`+6.13`** | **`+8.51`** | **`-2.04` (`-27.1%`)** | **`-2.79` (`-26.7%`)** | **`+0.1270`** |

### 2. Model 1: `BinaryGenderCNN`

- **Best Validation Epoch**: Epoch 12 (`Val Loss: 0.2510`, `Val Acc: 88.75%`)
- **Test Performance**: **`89.57%` Accuracy** | **`89.53%` Macro F1**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Male** | `0.8875` | `0.9166` | `0.9018` | `1,859` |
| **Female** | `0.9053` | `0.8727` | `0.8887` | `1,697` |

![BinaryGenderCNN Results](assets/05_cnn_binary_gender_results.png)

### 3. Model 2: `MultiClassRaceCNN`

- **Best Validation Epoch**: Epoch 10 (`Val Acc: 72.55%`, `Val Macro F1: 65.41%`)
- **Test Performance**: **`72.13%` Accuracy** | **`65.01%` Macro F1** | **`72.92%` Weighted F1**

| Class | Weight ($w_c$) | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **White** | `0.470` | `0.7831` | `0.8690` | `0.8238` | `1,512` |
| **Black** | `1.048` | `0.8028` | `0.7673` | `0.7846` | `679` |
| **Asian** | `1.381` | `0.8806` | `0.6155` | `0.7246` | `515` |
| **Indian** | `1.193` | `0.7732` | `0.5034` | `0.6098` | `596` |
| **Others** | `2.802` | `0.2349` | `0.4449` | `0.3075` | `254` |

![MultiClassRaceCNN Results](assets/06_cnn_multiclass_race_results.png)

### 4. Model 3: `AgeRegressionCNN`

- **Best Validation Epoch**: Epoch 15 (`Train MAE: 7.50` years, `Val MAE: 7.42` years)
- **Test Performance**: **`MAE = 7.53` years** | **`RMSE = 10.46` years** | **$R^2 = 0.7261$**

![AgeRegressionCNN Results](assets/07_cnn_age_regression_results.png)

### 5. Model 4: `MultiTaskFaceResNet` (`ResNet-18`)

Fine-tuned for 8 epochs (best checkpoint at Epoch 5 with `Val Loss = 1.1765`), the multi-task `ResNet-18` model achieves **`92.15%` Gender Accuracy**, **`73.52%` Race Macro F1**, and reduces **Age MAE to `5.49` years** ($R^2 = 0.8531$).

#### Sample Test Predictions (`MultiTaskFaceResNet`)
![Test Set Predictions](assets/08_multitask_resnet18_test_predictions.png)

#### Custom Image Inference
![Custom Image Inference](assets/09_custom_image_inference.png)

---

## Usage

1. Open `UTKFace_1.ipynb` (EDA) and `UTKFace_2.ipynb` (CNN Training) in **Google Colab**.
2. Set the hardware accelerator to **T4 GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU`).
3. Run all cells. The dataset is downloaded automatically via `kagglehub`.
4. Refer to [`REPORT.md`](REPORT.md) for detailed methodology and evaluation analysis.
