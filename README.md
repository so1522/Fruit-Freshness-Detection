# Fruit Freshness Detection System

An image-based machine learning web application that classifies fruit images into **fresh** and **rotten** categories. The system uses computer vision features such as color, texture, shape, and HOG features, and then applies machine learning models to identify the fruit and its freshness.

The project also includes a Flask backend and a browser-based dashboard where users can upload an image, use the camera, or test the included sample images.

---

## 1. Project Objective

The main objective is to automate basic fruit freshness inspection from an image.

The system:

- Takes a fruit image as input.
- Preprocesses the image.
- Extracts visual features.
- Classifies the image into one of six classes.
- Shows the prediction confidence.
- Displays class probabilities.
- Provides basic storage and consumption guidance.
- Shows selected image features through the web dashboard.

---

## 2. Supported Classes

The trained model supports six classes:

1. Fresh Apple
2. Fresh Banana
3. Fresh Orange
4. Rotten Apple
5. Rotten Banana
6. Rotten Orange

The labels used by the trained model are:

```text
fresh apple
fresh banana
fresh orange
rottenapples
rottenbanana
rottenoranges
```

---

## 3. Dataset

The training notebook used **9,199 images** distributed across the six classes.

| Class | Images |
|---|---:|
| Fresh Apple | 1,599 |
| Fresh Banana | 1,544 |
| Fresh Orange | 1,564 |
| Rotten Apple | 1,482 |
| Rotten Banana | 1,529 |
| Rotten Orange | 1,481 |
| **Total** | **9,199** |

The data was divided using a stratified **80:20 train-test split**:

- Training images: **7,359**
- Testing images: **1,840**

Stratification was used so that the class distribution remained similar in both sets.

---

## 4. Machine Learning Pipeline

The complete pipeline is:

```text
Input Image
     ↓
Resize to 128 × 128
     ↓
Feature Extraction
     ↓
Color Features
     ├── RGB
     ├── HSV
     └── LAB
     ↓
Texture Features
     ├── Laplacian Variance
     ├── GLCM
     └── Grayscale Entropy
     ↓
Shape Features
     ├── Contour Area
     ├── Perimeter
     ├── Circularity
     ├── Solidity
     ├── Aspect Ratio
     └── Extent
     ↓
Decay Feature
     └── Dark Pixel Ratio
     ↓
HOG Features
     ↓
8130-Dimensional Feature Vector
     ↓
Train/Test Split
     ↓
Standard Scaling for scale-sensitive models
     ↓
Model Training & Comparison
     ↓
XGBoost Selected for Deployment
     ↓
Prediction + Confidence
     ↓
Web Dashboard
```

---

## 5. Feature Extraction

Each image is resized to **128 × 128 pixels** before feature extraction.

The project combines several types of handcrafted computer vision features.

### Color Features

The system extracts statistical information from:

- RGB
- HSV
- LAB

For example, it calculates mean and standard deviation values for the color channels.

For HSV Hue, circular statistics are used because Hue is a circular value.

### Texture Features

The system extracts:

- Laplacian variance
- GLCM contrast
- GLCM energy
- GLCM homogeneity
- Grayscale entropy

These features help capture surface patterns and visual changes that can occur during spoilage.

### Shape Features

The largest detected contour is used to calculate:

- Contour area
- Perimeter
- Circularity
- Solidity
- Aspect ratio
- Extent

### Decay Feature

The system calculates the **dark pixel ratio**, which represents the proportion of dark pixels in the image. It can provide an additional visual signal for dark spots or decay.

### HOG Features

Histogram of Oriented Gradients (HOG) is used to capture local edge and shape information.

The final feature vector contains:

- 30 scalar features
- 8,100 HOG features
- **8,130 features in total**

---

## 6. Models Compared

Five classification algorithms were trained and evaluated:

- K-Nearest Neighbors (KNN)
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- XGBoost

### Test Results

The results recorded in the training notebook were:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| KNN | 64.62% | 69.46% | 64.62% | 65.47% |
| Random Forest | 84.62% | 84.49% | 84.62% | 84.52% |
| SVM | 84.84% | 84.81% | 84.84% | 84.80% |
| Logistic Regression | 80.38% | 80.49% | 80.38% | 80.42% |
| **XGBoost** | **93.86%** | **93.88%** | **93.86%** | **93.83%** |

Based on this recorded test evaluation, the saved deployment model is **XGBoost**.

The reported XGBoost test accuracy is **93.86%** on the 1,840-image test set.

---

## 7. XGBoost Configuration

The trained XGBoost model uses:

```text
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
eval_metric = mlogloss
random_state = 42
```

XGBoost was trained directly on the extracted feature vectors.

---

## 8. Classification Report

The recorded XGBoost test evaluation was:

| Class | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Fresh Apple | 0.95 | 0.98 | 0.96 |
| Fresh Banana | 1.00 | 0.99 | 0.99 |
| Fresh Orange | 0.89 | 0.94 | 0.91 |
| Rotten Apple | 0.91 | 0.86 | 0.89 |
| Rotten Banana | 0.98 | 0.98 | 0.98 |
| Rotten Orange | 0.91 | 0.88 | 0.89 |

Overall accuracy: **0.94 / 93.86%**

---

## 9. Project Structure

```text
Fruit-Freshness-Detection/
│
├── backend/
│   ├── app.py
│   ├── feature_extractor.py
│   ├── model_loader.py
│   └── run.py
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── saved_models/
│   ├── best_model.pkl
│   ├── best_model_name.pkl
│   ├── label_encoder.pkl
│   └── scaler.pkl
│
├── ffd.ipynb
├── requirements.txt
├── test-image.jpg
├── fa.jpeg
├── fo.jpeg
├── ro.jpeg
├── bb.jpeg
├── ra.jpeg
├── ro1jpeg.jpeg
├── .gitignore
└── README.md
```

### Important Files

**`ffd.ipynb`**

Contains the model development workflow:

- Feature extraction
- Dataset loading
- Label encoding
- Train-test split
- Scaling
- Model training
- Model comparison
- Classification report
- Confusion matrix
- Model saving
- Test-image prediction

**`backend/feature_extractor.py`**

Contains the feature extraction logic used by the web application. It must remain consistent with the features used during model training.

**`backend/model_loader.py`**

Loads the saved model, scaler, and label encoder and performs predictions.

**`backend/app.py`**

Flask API that handles image uploads, sample-image predictions, health checks, and frontend serving.

**`backend/run.py`**

Starts the Flask application and opens the web application in the default browser.

**`frontend/index.html`**

Main user interface.

**`frontend/js/app.js`**

Handles image upload, camera input, API requests, charts, and result updates.

**`saved_models/**`**

Contains the trained XGBoost model and preprocessing objects required for inference.

---

## 10. Web Application

The project provides a browser-based dashboard.

Users can:

- Upload an image.
- Drag and drop an image.
- Use the device camera.
- Select a sample image.
- View the predicted fruit.
- View fresh/rotten status.
- View prediction confidence.
- View class probabilities.
- View selected color/texture/shape metrics.
- Get basic storage guidance.

The frontend communicates with the Flask backend through REST API endpoints.

---

## 11. API Endpoints

### Health Check

```http
GET /api/health
```

Returns the API status, model name, and supported classes.

### Sample Images

```http
GET /api/samples
```

Returns the available sample images.

### Sample Image

```http
GET /api/sample-image/<filename>
```

Returns a sample image.

### Prediction

```http
POST /api/predict
```

The endpoint accepts:

1. An uploaded image file using `multipart/form-data`.
2. A base64 encoded image using JSON.
3. A project sample image using JSON.

Example file upload:

```text
POST /api/predict
Content-Type: multipart/form-data
file=<image>
```

A successful response contains information such as:

```json
{
  "prediction": "fresh apple",
  "display_name": "Fresh Apple",
  "fruit_type": "Apple",
  "is_fresh": true,
  "confidence": 95.2,
  "probabilities": {},
  "advice": {},
  "metrics": {},
  "model_used": "XGBoost"
}
```

---

## 12. Installation

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
cd Fruit-Freshness-Detection
```

### Step 2: Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

The backend uses Flask, so if Flask is not already installed in your environment:

```bash
pip install flask
```

For reproducibility, it is recommended to use the versions recorded in `requirements.txt`, especially the versions of `scikit-learn` and `xgboost` used when the saved model was created.

---

## 13. Run the Application

From the project root:

```bash
python run.py
```

The application will start at:

```text
http://127.0.0.1:5000
```

The health endpoint is:

```text
http://127.0.0.1:5000/api/health
```

`run.py` also attempts to open the web application automatically in the default browser.

---

## 14. How Prediction Works

When a user uploads an image:

```text
Image Upload
     ↓
Flask receives image
     ↓
OpenCV decodes image
     ↓
Image resized to 128 × 128
     ↓
Feature extraction
     ↓
8,130 features generated
     ↓
XGBoost model receives features
     ↓
Class prediction
     ↓
Prediction probabilities
     ↓
Result formatted by ModelLoader
     ↓
JSON response
     ↓
Frontend displays result
```

The same feature extraction logic used during model development is implemented in `backend/feature_extractor.py`, which helps keep training and inference consistent.

---

## 15. Saved Model Files

### `best_model.pkl`

Contains the trained XGBoost classifier.

### `best_model_name.pkl`

Stores:

```text
XGBoost
```

### `label_encoder.pkl`

Stores the mapping between numerical labels and fruit classes.

### `scaler.pkl`

Stores the `StandardScaler` used for scale-sensitive models during experimentation and retained with the project artifacts.

For the deployed XGBoost model, the backend does not apply the scaler before prediction because the notebook trained XGBoost on the original feature vectors.

---

## 16. Example Prediction

For the included test image, the notebook recorded:

```text
Predicted Class: rottenoranges
Confidence: 99.45%
```

Recorded class probabilities included:

```text
fresh apple:     0.07%
fresh banana:    0.01%
fresh orange:    0.25%
rottenapples:    0.21%
rottenbanana:    0.02%
rottenoranges:  99.45%
```

These values are from the saved notebook experiment and should not be treated as a guarantee for every new image.

---

## 17. Technologies Used

### Programming

- Python
- JavaScript
- HTML
- CSS

### Machine Learning

- XGBoost
- Scikit-learn
- KNN
- Random Forest
- SVM
- Logistic Regression

### Computer Vision

- OpenCV
- HOG
- GLCM
- RGB
- HSV
- LAB
- Contour-based shape analysis

### Scientific Computing

- NumPy
- SciPy
- Pandas

### Backend

- Flask

### Frontend

- HTML5
- CSS3
- JavaScript
- Chart.js
- Font Awesome

### Model Persistence

- Joblib

---

## 18. Challenges Solved

### Different visual characteristics

Fresh and rotten produce can have similar colors and shapes. To capture more information, the project combines color, texture, shape, decay, and HOG features instead of relying on a single feature type.

### High-dimensional feature vector

HOG produces a large number of features. The final feature vector contains 8,130 values. Tree-based models such as XGBoost were evaluated to handle this feature representation.

### Comparing different ML algorithms

Instead of assuming one algorithm would work best, five classifiers were trained and evaluated using the same train-test split.

### Consistent inference

The feature extraction logic used by the backend follows the same 128 × 128 preprocessing and feature construction used in the training notebook.

### User-friendly prediction

A Flask API and web dashboard were added so that the trained model can be used through image upload or camera input instead of requiring users to run notebook cells.

---

## 19. Limitations

This project is a machine learning prototype and has some limitations:

- It supports only six trained classes.
- Prediction quality depends on image quality and similarity to the training data.
- Different lighting, backgrounds, camera angles, and occlusion can affect predictions.
- The model does not perform laboratory-level food safety or microbiological testing.
- A visual "fresh" prediction should not be treated as a guarantee that food is safe to eat.
- The training dataset and test set are image-based, so real-world performance may differ from the recorded test accuracy.

---

## 20. Future Improvements

Possible improvements include:

- Add more fruit and vegetable categories.
- Increase dataset size and diversity.
- Add images from different lighting and background conditions.
- Use data augmentation during training.
- Perform systematic hyperparameter tuning.
- Add cross-validation and more robust external testing.
- Add explainability using SHAP or similar methods.
- Improve segmentation so the model focuses on the produce rather than the background.
- Add cloud deployment using Flask/FastAPI.
- Add a database for prediction history.
- Add batch image processing.
- Add mobile support.
- Add monitoring for model performance after deployment.

---

## 21. Running the Training Notebook

If you want to retrain the model:

```bash
jupyter notebook
```

Open:

```text
ffd.ipynb
```

The notebook performs:

```text
Import libraries
    ↓
Define feature extraction
    ↓
Load dataset
    ↓
Extract features
    ↓
Encode labels
    ↓
Train-test split
    ↓
Feature scaling
    ↓
Train five ML models
    ↓
Compare results
    ↓
Select XGBoost
    ↓
Generate classification report
    ↓
Generate confusion matrix
    ↓
Save model
    ↓
Test a new image
```

The dataset path inside the notebook is currently configured as a local Windows path, so it should be changed to the dataset location on the machine where retraining is performed.

---

## 22. Important Reproducibility Note

The repository contains the trained model files, so retraining is not required just to run the web application.

If you retrain the model, keep the following consistent between training and inference:

- Image resize: `128 × 128`
- Feature extraction order
- HOG configuration
- Feature count
- Label mapping
- Model type
- Required preprocessing

The saved model was created with the project's recorded dependency versions. Using significantly different versions of model-persistence libraries can produce compatibility warnings or unexpected behavior.

---
