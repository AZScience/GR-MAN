@echo off
title He Thong AI Du Bao Bien Co Lam Sang MIMIC-IV
echo ===================================================================
echo     DANG KHOI DONG HE THONG DU BAO BIEN CO Y TE DA PHUONG THUC
echo ===================================================================
cd /d "%~dp0"
streamlit run app_medical_pipeline.py --server.port 8501
pause
