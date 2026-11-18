import os
import gzip
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data", "mimic_iv_demo")

class RealMIMICDataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.hosp_dir = os.path.join(data_dir, "hosp")
        self.icu_dir = os.path.join(data_dir, "icu")
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=64, stop_words='english')

    def load_and_preprocess(self):
        print("[*] Đang tải các bảng dữ liệu thực tế từ MIMIC-IV Demo...")
        # 1. Admissions & Patients
        df_adm = pd.read_csv(os.path.join(self.hosp_dir, "admissions.csv.gz"), compression='gzip')
        df_pat = pd.read_csv(os.path.join(self.hosp_dir, "patients.csv.gz"), compression='gzip')
        df_diag = pd.read_csv(os.path.join(self.hosp_dir, "diagnoses_icd.csv.gz"), compression='gzip')
        df_d_diag = pd.read_csv(os.path.join(self.hosp_dir, "d_icd_diagnoses.csv.gz"), compression='gzip')
        df_labs = pd.read_csv(os.path.join(self.hosp_dir, "labevents.csv.gz"), compression='gzip')

        # Ghép thông tin bệnh nhân
        df_base = df_adm[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type', 'hospital_expire_flag']].copy()
        df_base = df_base.merge(df_pat[['subject_id', 'gender', 'anchor_age']], on='subject_id', how='left')

        # Chuyển đổi giới tính thành số (1: Male, 0: Female)
        df_base['is_male'] = (df_base['gender'] == 'M').astype(float)
        
        # One-hot cho loại nhập viện
        adm_type_dummies = pd.get_dummies(df_base['admission_type'], prefix='adm', drop_first=True).astype(float)
        
        # 2. Xử lý Phương thức Văn bản lâm sàng (Clinical Text Modality) từ chẩn đoán ICD-10
        print("[*] Trích xuất phương thức văn bản lâm sàng từ bảng chẩn đoán ICD...")
        df_diag_merged = df_diag.merge(df_d_diag, on=['icd_code', 'icd_version'], how='left')
        df_text_grouped = df_diag_merged.groupby('hadm_id')['long_title'].apply(
            lambda x: ' '.join(x.dropna().astype(str).tolist())
        ).reset_index()
        df_text_grouped.rename(columns={'long_title': 'clinical_text'}, inplace=True)
        
        df_base = df_base.merge(df_text_grouped, on='hadm_id', how='left')
        df_base['clinical_text'] = df_base['clinical_text'].fillna("No explicit diagnostic note recorded")

        # 3. Xử lý Phương thức Xét nghiệm sinh hóa & Dấu hiệu chuyển hóa (Labs Modality)
        print("[*] Trích xuất các chỉ số xét nghiệm lâm sàng (Labs: Creatinine, Potassium, Sodium, WBC, Platelets, BUN)...")
        # Danh sách mã lab quan trọng:
        # 50912: Creatinine, 50971: Potassium, 50983: Sodium, 51006: Urea Nitrogen (BUN), 51265: Platelet, 51222: Hemoglobin
        target_labs = {
            50912: 'creatinine',
            50971: 'potassium',
            50983: 'sodium',
            51006: 'bun',
            51265: 'platelet',
            51222: 'hemoglobin'
        }
        labs_filtered = df_labs[df_labs['itemid'].isin(target_labs.keys())][['hadm_id', 'itemid', 'valuenum']].dropna()
        labs_filtered['lab_name'] = labs_filtered['itemid'].map(target_labs)
        labs_pivot = labs_filtered.pivot_table(index='hadm_id', columns='lab_name', values='valuenum', aggfunc='mean').reset_index()

        df_base = df_base.merge(labs_pivot, on='hadm_id', how='left')

        # Ghép đặc trưng dạng số
        lab_cols = list(target_labs.values())
        num_cols = ['anchor_age', 'is_male'] + lab_cols + list(adm_type_dummies.columns)
        
        df_full = pd.concat([df_base, adm_type_dummies], axis=1)

        # Điền giá trị khuyết thiếu (Missing value imputation) bằng trung vị thực tế (Median Imputation)
        for col in lab_cols:
            med_val = df_full[col].median()
            df_full[col] = df_full[col].fillna(med_val if not np.isnan(med_val) else 0.0)
        df_full['anchor_age'] = df_full['anchor_age'].fillna(df_full['anchor_age'].median())
        df_full['is_male'] = df_full['is_male'].fillna(0.5)

        # Vector hóa văn bản bằng TF-IDF
        text_matrix = self.tfidf.fit_transform(df_full['clinical_text']).toarray() # shape: (N, 64)

        # Chuẩn hóa đặc trưng dạng số
        num_matrix = self.scaler.fit_transform(df_full[num_cols].values)

        labels = df_full['hospital_expire_flag'].values.astype(np.float32)

        print(f"[+] Hoàn tất tiền xử lý: Tổng số mẫu = {len(df_full)}, Số ca biến cố tử vong = {int(labels.sum())}/{len(labels)}")
        
        return {
            'df_raw': df_full,
            'num_features': num_matrix,
            'num_feature_names': num_cols,
            'text_features': text_matrix,
            'labels': labels,
            'hadm_ids': df_full['hadm_id'].values,
            'subject_ids': df_full['subject_id'].values,
            'clinical_texts': df_full['clinical_text'].values
        }

# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH HỌC SÂU ĐA PHƯƠNG THỨC THỰC TẾ
# ==============================================================================
class RealMultimodalFusionModel(nn.Module):
    def __init__(self, num_dim, text_dim=64, hidden_dim=48):
        super(RealMultimodalFusionModel, self).__init__()
        
        # Nhánh 1: Trích xuất đặc trưng có cấu trúc (Demographics + Labs)
        self.num_encoder = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Nhánh 2: Trích xuất đặc trưng văn bản lâm sàng (Clinical Diagnostic Text)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Nhánh 3: Tương tác & Dung hợp Đa phương thức (Cross-Modal Attention Fusion)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Đầu ra dự báo sớm biến cố (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, num_x, text_x):
        # Mã hóa nhánh 1
        h_num = self.num_encoder(num_x) # (B, hidden_dim)
        # Mã hóa nhánh 2
        h_text = self.text_encoder(text_x) # (B, hidden_dim)
        
        # Đưa vào Cross-Attention: tạo 2 tokens biểu diễn 2 phương thức
        tokens = torch.stack([h_num, h_text], dim=1) # (B, 2, hidden_dim)
        attn_out, attn_weights = self.cross_attention(tokens, tokens, tokens)
        
        fused = attn_out.reshape(attn_out.size(0), -1) # (B, hidden_dim * 2)
        logits = self.classifier(fused)
        return logits, attn_weights

class GatedResidualMultimodalModel(nn.Module):
    """
    Kiến trúc Học Sâu Đa Phương Thức Tiên Tiến: GR-MAN (Gated Residual Multimodal Attention Network)
    Tối ưu chuyên biệt cho Dự Báo Sớm Các Biến Cố Bất Lợi Trong Chăm Sóc Sức Khỏe (Early Warning CDSS).
    Tích hợp Cổng Thích Ứng GMU và Đường Truyền Tắt Định Danh (Residual Identity Highway).
    """
    def __init__(self, num_dim, text_dim=64, hidden_dim=48):
        super(GatedResidualMultimodalModel, self).__init__()
        
        # Nhánh 1: Trích xuất đặc trưng có cấu trúc (Demographics + 6 Labs)
        self.num_encoder = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Nhánh 2: Trích xuất đặc trưng văn bản lâm sàng (Clinical Diagnostic Text)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Nhánh 3: Tương tác & Dung hợp Đa phương thức (Cross-Modal Attention)
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)
        
        # Nhánh 4: Cổng Thích Ứng Đa Phương Thức (Gated Multimodal Unit - GMU)
        # g = σ(W_g · [h_num || Context] + b_g)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Nhánh 5: Đầu ra dự báo sớm biến cố y tế (Classification Head)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, num_x, text_x):
        h_num = self.num_encoder(num_x)
        h_text = self.text_encoder(text_x)
        
        # Đưa vào Cross-Attention: 2 tokens đại diện cho 2 phương thức
        tokens = torch.stack([h_num, h_text], dim=1) # (B, 2, hidden_dim)
        attn_out, attn_weights = self.cross_attention(tokens, tokens, tokens)
        
        # Context biểu diễn tương quan lâm sàng
        context = attn_out[:, 1, :] # (B, hidden_dim)
        
        # Cổng thích ứng g ∈ [0, 1]
        gate_inp = torch.cat([h_num, context], dim=-1) # (B, hidden_dim * 2)
        g = self.gate(gate_inp) # (B, hidden_dim)
        
        # Dung hợp có cổng kết hợp đường truyền tắt Residual Highway
        fused = g * h_num + (1.0 - g) * context
        final_rep = torch.cat([h_num, fused], dim=-1) # (B, hidden_dim * 2)
        logits = self.classifier(final_rep)
        
        return logits, attn_weights, g

# ==============================================================================
# 3. HUẤN LUYỆN VÀ ĐÁNH GIÁ THỰC TẾ 100%
# ==============================================================================
def train_and_evaluate_real_pipeline():
    loader = RealMIMICDataLoader()
    data = loader.load_and_preprocess()

    num_features = data['num_features']
    text_features = data['text_features']
    labels = data['labels']
    indices = np.arange(len(labels))

    # Chia Train/Test có phân tầng (Stratified) dựa trên nhãn biến cố thật
    idx_train, idx_test = train_test_split(indices, test_size=0.25, random_state=42, stratify=labels)

    # Chuyển sang PyTorch Tensor
    X_num_train = torch.tensor(num_features[idx_train], dtype=torch.float32)
    X_text_train = torch.tensor(text_features[idx_train], dtype=torch.float32)
    y_train = torch.tensor(labels[idx_train], dtype=torch.float32).unsqueeze(1)

    X_num_test = torch.tensor(num_features[idx_test], dtype=torch.float32)
    X_text_test = torch.tensor(text_features[idx_test], dtype=torch.float32)
    y_test = torch.tensor(labels[idx_test], dtype=torch.float32).unsqueeze(1)

    model = RealMultimodalFusionModel(num_dim=num_features.shape[1], text_dim=text_features.shape[1])
    
    # Tính trọng số phạt mất cân bằng nhãn thực tế
    pos_count = labels.sum()
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor([neg_count / max(1.0, pos_count)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    print("\n[*] Bắt đầu huấn luyện mô hình học sâu đa phương thức trên dữ liệu thật...")
    model.train()
    for epoch in range(1, 41):
        optimizer.zero_grad()
        logits, _ = model(X_num_train, X_text_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"    Epoch {epoch:02d}/40 | BCE Loss: {loss.item():.4f}")

    # ĐÁNH GIÁ TRÊN TẬP TEST THẬT
    model.eval()
    with torch.no_grad():
        test_logits, test_attn = model(X_num_test, X_text_test)
        test_probs = torch.sigmoid(test_logits).squeeze().cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        y_true = labels[idx_test]

    auroc = roc_auc_score(y_true, test_probs)
    auprc = average_precision_score(y_true, test_probs)
    f1 = f1_score(y_true, test_preds, zero_division=0)
    cm = confusion_matrix(y_true, test_preds)

    print("\n" + "=" * 65)
    print(" KẾT QUẢ ĐÁNH GIÁ THỰC TẾ TRÊN TẬP KIỂM THỬ MIMIC-IV (TEST SET)")
    print("=" * 65)
    print(f"  • AUROC (Area Under ROC Curve)        : {auroc:.4f}")
    print(f"  • AUPRC (Area Under PR Curve)         : {auprc:.4f}")
    print(f"  • F1-Score                            : {f1:.4f}")
    print(f"  • Ma trận nhầm lẫn (Confusion Matrix):\n{cm}")
    print("=" * 65)

    # Lưu model và dataset test thực tế
    checkpoint_path = os.path.join(WORKSPACE_DIR, "real_multimodal_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_feature_names': data['num_feature_names'],
        'scaler': loader.scaler,
        'tfidf': loader.tfidf,
        'metrics': {'auroc': auroc, 'auprc': auprc, 'f1': f1, 'cm': cm.tolist()},
        'idx_test': idx_test,
        'data': data
    }, checkpoint_path)
    print(f"[✓] Đã lưu mô hình huấn luyện thực tế vào: {checkpoint_path}")

if __name__ == "__main__":
    train_and_evaluate_real_pipeline()
