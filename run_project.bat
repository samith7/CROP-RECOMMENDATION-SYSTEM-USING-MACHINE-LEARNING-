@echo off
echo ==========================================
echo Crop Recommendation ML Mini Project
echo ==========================================
echo.
python -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model.py
streamlit run app.py
pause
