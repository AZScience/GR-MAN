import os
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_eicu_cohort(data_dir=os.path.join(WORKSPACE_DIR, "data", "eicu_demo")):
    processed_pt_path = os.path.join(WORKSPACE_DIR, "eicu_cohort_processed.pt")
    if os.path.exists(processed_pt_path):
        print(f"[*] Nạp trực tiếp đoàn hệ eICU đã xử lý từ checkpoint {processed_pt_path}...")
        return torch.load(processed_pt_path, map_location=torch.device('cpu'), weights_only=False)

    print("[*] Đang tải đoàn hệ lớn eICU từ tệp tin CSV thô (2,520 bệnh nhân ICU đa trung tâm)...")
    pat_file = os.path.join(data_dir, "patient.csv.gz")
    if not os.path.exists(pat_file):
        raise FileNotFoundError(f"Không tìm thấy dữ liệu eICU tại: {data_dir} hoặc {processed_pt_path}")

    pat = pd.read_csv(pat_file, compression='gzip')
    diag = pd.read_csv(os.path.join(data_dir, "diagnosis.csv.gz"), compression='gzip')
    labs = pd.read_csv(os.path.join(data_dir, "lab.csv.gz"), compression='gzip')

    # Nhãn tử vong thực tế
    pat['hospital_expire_flag'] = (pat['hospitaldischargestatus'] == 'Expired').astype(float)
    pat['age_clean'] = pd.to_numeric(pat['age'].replace('> 89', '90'), errors='coerce').fillna(62.0)
    pat['is_male'] = (pat['gender'] == 'Male').astype(float)
    pat['los_days'] = pd.to_numeric(pat['unitdischargeoffset'], errors='coerce').fillna(1440.0) / 1440.0

    # Chẩn đoán văn bản
    diag_texts = diag.groupby('patientunitstayid')['diagnosisstring'].apply(
        lambda x: '; '.join(x.dropna().unique().tolist()[:10])
    ).reset_index()
    diag_texts.rename(columns={'diagnosisstring': 'clinical_text'}, inplace=True)
    pat = pat.merge(diag_texts, on='patientunitstayid', how='left')
    pat['clinical_text'] = pat['clinical_text'].fillna("General ICU admission without detailed diagnostic string")

    # Các xét nghiệm máu quan trọng
    target_labs = {
        'creatinine': 'creatinine',
        'BUN': 'bun',
        'potassium': 'potassium',
        'sodium': 'sodium',
        'WBC x 1000': 'wbc',
        'platelets x 1000': 'platelet',
        'glucose': 'glucose'
    }
    labs_f = labs[labs['labname'].isin(target_labs.keys())][['patientunitstayid', 'labname', 'labresult']].dropna()
    labs_f['lab_norm'] = labs_f['labname'].map(target_labs)
    labs_piv = labs_f.pivot_table(index='patientunitstayid', columns='lab_norm', values='labresult', aggfunc='mean').reset_index()

    df_full = pat.merge(labs_piv, on='patientunitstayid', how='left')
    
    num_cols = ['age_clean', 'is_male', 'los_days', 'creatinine', 'bun', 'potassium', 'sodium', 'wbc', 'platelet', 'glucose']
    for c in ['creatinine', 'bun', 'potassium', 'sodium', 'wbc', 'platelet', 'glucose']:
        df_full[c] = df_full[c].fillna(df_full[c].median())
        
    df_full['subject_id'] = df_full['uniquepid']
    df_full['hadm_id'] = df_full['patientunitstayid']
    df_full['anchor_age'] = df_full['age_clean']
    df_full['gender'] = df_full['gender'].fillna('Unknown')
    df_full['admission_type'] = df_full['unitstaytype'].fillna('ICU')

    scaler = StandardScaler()
    num_matrix = scaler.fit_transform(df_full[num_cols].values)

    tfidf = TfidfVectorizer(max_features=64, stop_words='english')
    text_matrix = tfidf.fit_transform(df_full['clinical_text']).toarray()

    labels = df_full['hospital_expire_flag'].values.astype(np.float32)

    print(f"[+] Hoàn tất nạp eICU: {len(df_full)} bệnh nhân, {int(labels.sum())} ca tử vong nội trú.")
    return {
        'df_raw': df_full,
        'num_features': num_matrix,
        'num_feature_names': num_cols,
        'text_features': text_matrix,
        'labels': labels,
        'scaler': scaler,
        'tfidf': tfidf,
        'dataset_name': "eICU Multi-Center Cohort (2,520 bệnh nhân, 212 ca tử vong)"
    }
