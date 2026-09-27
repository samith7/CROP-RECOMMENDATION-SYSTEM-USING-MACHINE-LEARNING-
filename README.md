# Crop Recommendation System Using Machine Learning

## Project
A machine-learning based system that recommends a suitable crop from soil and environmental conditions.

### Inputs
- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature (°C)
- Humidity (%)
- Soil pH
- Rainfall (mm)

### Algorithms
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Random Forest is used for the final prediction after comparing the models on the test set.

## Dataset
The project uses the commonly used Crop Recommendation dataset containing 7 input features and crop labels. The training script automatically downloads the CSV from a public GitHub mirror if the dataset is not present locally.

Source mirror:
https://github.com/nileshiq/Crop-Recommendation

The repository describes a 2,200-row dataset with Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH_Value, Rainfall and Crop fields.

## VS Code Setup

### 1. Open the project
Extract the ZIP and open the extracted folder in VS Code.

### 2. Open terminal
VS Code -> Terminal -> New Terminal

### 3. Create virtual environment
Windows:
```powershell
python -m venv venv
venv\Scripts\activate
```

If PowerShell blocks activation:
```powershell
venv\Scripts\activate.bat
```

### 4. Install libraries
```powershell
pip install -r requirements.txt
```

### 5. Train the models
```powershell
python train_model.py
```

This downloads the dataset if required, preprocesses it, trains four ML algorithms, compares their performance, saves the Random Forest model, and creates result files.

### 6. Run the application
```powershell
streamlit run app.py
```

A browser window should open. If it does not, open the localhost URL shown in the terminal.

## Project Structure

```text
Crop_Recommendation_ML_Project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── Crop_Recommendation.csv   # downloaded automatically
├── models/
│   ├── crop_model.pkl
│   └── feature_scaler.pkl
└── results/
    ├── model_comparison.csv
    ├── classification_report.txt
    ├── confusion_matrix.png
    └── feature_importance.png
```

## Important
Run `train_model.py` before starting Streamlit. Do not claim that the model is production-ready. This is an academic mini project based on a public crop-recommendation dataset.
