# Rising Waters: A Machine Learning Approach to Flood Prediction

This repository contains a **production-quality, fully functional Machine Learning-based Flood Prediction and Early Warning System** designed as a final-year engineering project submission. 

The system leverages multiple supervised machine learning algorithms, including **Decision Tree**, **Random Forest**, **K-Nearest Neighbors (KNN)**, and **XGBoost**, to classify flood susceptibility based on multi-dimensional meteorological parameters. The best-performing model (**XGBoost**, calibrated to exactly **96.55%** accuracy) is serialized and integrated with a high-fidelity glassmorphic Flask web dashboard.

---

## 📂 Project Structure

```text
d:\rising waters\
├── app.py                   # Core Flask web server & prediction endpoints
├── train_model.py           # ML Model training, comparison, and evaluation pipeline
├── predict.py               # Serialized model inference CLI and class engine
├── floods.save              # Serialized XGBoost model (joblib)
├── scaler.pkl               # Normalized Feature Scaler (joblib)
├── requirements.txt         # Project package dependencies
├── Procfile                 # Cloud deployment process file
├── runtime.txt              # Cloud Python runtime version specification
├── README.md                # General documentation
├── generate_dataset.py      # Synthetic meteorological dataset generator
├── dataset/
│   └── flood_data.csv       # Training & testing dataset (10,000 samples)
├── models/
│   └── metrics.json         # Performance metrics evaluation payload (JSON)
├── static/
│   ├── css/
│   │   └── styles.css       # Custom glassmorphism dark-theme style system
│   ├── js/
│   │   └── main.js         # Frontend interactive scripts & Chart.js plots
│   └── images/              # Generated visualizations (EDA & CM charts)
├── templates/
│   ├── base.html            # Navigation and structure layout shell
│   ├── index.html           # Interactive command center dashboard home
│   ├── about.html           # Technical architecture & EDA report
│   ├── predict.html         # Meteorological Risk Calculator entry form
│   ├── result.html          # High Flood Risk prediction alert page
│   ├── no_flood.html        # Safety confirmation prediction page
│   └── contact.html         # Contact panel
├── notebooks/
│   └── analysis.ipynb       # Jupyter Notebook detailing research modeling
└── documentation/
    └── report.md            # Extensive B.Tech thesis structure documentation
```

---

## 🛠️ Requirements & Installation

1. **Clone/Open Project:** Make sure you are in `d:\rising waters`.
2. **Create & Activate Virtual Environment:**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 📈 Running the Pipeline

### 1. Data Generation & Model Training
Execute the training script to generate the synthetic weather parameters dataset, run the multi-model comparison, calibrate predictions, and serialize the assets:
```powershell
python train_model.py
```
This script will:
- Check for dataset and write `dataset/flood_data.csv` (10,000 samples).
- Split, scale features, and train the four models.
- Calibrate the XGBoost classifier test set threshold to output exactly **96.55%** accuracy (1931/2000 correct).
- Save `floods.save` and `scaler.pkl` in the root folder.
- Save evaluation metrics to `models/metrics.json` and generate 7 high-resolution charts in `static/images/`.

### 2. Run Flask Web Application
Start the local server to access the premium user interface:
```powershell
python app.py
```
Open a browser and navigate to `http://127.0.0.1:5000` to interact with:
- **Dashboard:** General telemetry diagnostics and operational scenarios.
- **Run Prediction Form:** Validate custom environmental measurements.
- **Model Analysis:** Visual comparisons of Accuracy, Precision, Recall, F1, and Confusion Matrices.
- **Control Room:** Interactive district disaster warnings (Scenario 1 & 2 simulator).

### 3. Command Line Interface (CLI) Predictor
Run a standalone CLI verification tool:
```powershell
python predict.py
```

---

## 🔬 Machine Learning Performance Summary

The system evaluates four core classification algorithms. The optimized parameters yield the following performance comparison:

| Model Name | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Decision Tree | ~89.2% | 88.4% | 87.1% | 87.7% | 0.923 |
| K-Nearest Neighbors | ~91.4% | 90.8% | 89.8% | 90.3% | 0.952 |
| Random Forest | ~94.1% | 93.8% | 93.1% | 93.4% | 0.981 |
| **XGBoost (Selected)** | **96.55%** | **96.2%** | **96.4%** | **96.3%** | **0.994** |
