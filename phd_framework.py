import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score, brier_score_loss, confusion_matrix
)
from real_data_pipeline import RealMIMICDataLoader, RealMultimodalFusionModel

# ==============================================================================
# 1. HÀM MẤT MÁT CHUYÊN BIỆT Y SINH: FOCAL LOSS VÀ WEIGHTED ASYMMETRIC LOSS
# ==============================================================================
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., ICCV 2017) giải quyết triệt để mất cân bằng nhãn y sinh:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Tập trung gradient vào các ca bệnh khó (hard negative/positive), giảm phạt các ca dễ.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # p_t: xác suất phân loại đúng
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ==============================================================================
# 2. ĐỘ HIỆU CHUẨN XÁC SUẤT LÂM SÀNG (CALIBRATION: EXPECTED CALIBRATION ERROR - ECE)
# ==============================================================================
def calculate_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE) - Thước đo độ tin cậy của xác suất dự báo lâm sàng.
    Trong y tế, một mô hình dự báo xác suất 80% thì thực tế 100 bệnh nhân phải có 80 người mắc bệnh.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if np.sum(in_bin) > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_data.append({
                'bin': f"[{bin_lower:.1f}-{bin_upper:.1f}]",
                'confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'count': int(np.sum(in_bin))
            })
    return float(ece), bin_data

# ==============================================================================
# 3. KHOẢNG TIN CẬY BOOTSTRAP 95% (95% CONFIDENCE INTERVAL - CI)
# ==============================================================================
def bootstrap_metric_ci(y_true, y_prob, metric_fn, n_bootstraps=500, alpha=0.05):
    """
    Tính khoảng tin cậy 95% CI theo chuẩn bài báo y sinh quốc tế
    """
    bootstrapped_scores = []
    rng = np.random.RandomState(42)
    n = len(y_true)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_fn(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
    
    if len(bootstrapped_scores) == 0:
        return 0.0, 0.0
    sorted_scores = np.sort(bootstrapped_scores)
    lower = sorted_scores[int(alpha / 2 * len(sorted_scores))]
    upper = sorted_scores[int((1 - alpha / 2) * len(sorted_scores))]
    return float(lower), float(upper)

# ==============================================================================
# 4. HUẤN LUYỆN 5-FOLD STRATIFIED CROSS-VALIDATION VỚI EARLY STOPPING
# ==============================================================================
class PhDEarlyWarningTrainer:
    def __init__(self, data_dict, hyperparams=None):
        self.data_dict = data_dict
        self.num_features = data_dict['num_features']
        self.text_features = data_dict['text_features']
        self.labels = data_dict['labels']
        
        # Bảng siêu tham số chuẩn mực
        self.hp = hyperparams or {
            'hidden_dim': 64,
            'num_heads': 4,
            'dropout': 0.25,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'focal_alpha': 0.75,
            'focal_gamma': 2.0,
            'max_epochs': 50,
            'patience': 10,
            'batch_size': 32
        }

    def run_stratified_kfold(self, n_splits=5):
        print("=" * 70)
        print(f"[*] TIẾN HÀNH {n_splits}-FOLD STRATIFIED CROSS VALIDATION CHUẨN QUỐC TẾ")
        print(f"[*] Siêu tham số cấu hình: {self.hp}")
        print("=" * 70)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        oof_predictions = np.zeros(len(self.labels))
        oof_targets = np.zeros(len(self.labels))

        for fold, (train_idx, val_idx) in enumerate(skf.split(self.num_features, self.labels), 1):
            print(f"\n---> Huấn luyện Fold {fold}/{n_splits} (Train: {len(train_idx)} | Val: {len(val_idx)})")

            X_num_tr = torch.tensor(self.num_features[train_idx], dtype=torch.float32)
            X_text_tr = torch.tensor(self.text_features[train_idx], dtype=torch.float32)
            y_tr = torch.tensor(self.labels[train_idx], dtype=torch.float32).unsqueeze(1)

            X_num_va = torch.tensor(self.num_features[val_idx], dtype=torch.float32)
            X_text_va = torch.tensor(self.text_features[val_idx], dtype=torch.float32)
            y_va = torch.tensor(self.labels[val_idx], dtype=torch.float32).unsqueeze(1)

            model = RealMultimodalFusionModel(
                num_dim=self.num_features.shape[1],
                text_dim=self.text_features.shape[1],
                hidden_dim=self.hp['hidden_dim']
            )

            criterion = BinaryFocalLoss(alpha=self.hp['focal_alpha'], gamma=self.hp['focal_gamma'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.hp['lr'], weight_decay=self.hp['weight_decay'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

            best_val_loss = float('inf')
            patience_counter = 0
            best_state = None

            for epoch in range(1, self.hp['max_epochs'] + 1):
                model.train()
                optimizer.zero_grad()
                logits, _ = model(X_num_tr, X_text_tr)
                loss = criterion(logits, y_tr)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                optimizer.step()

                # Đánh giá validation
                model.eval()
                with torch.no_grad():
                    val_logits, _ = model(X_num_va, X_text_va)
                    val_loss = criterion(val_logits, y_va).item()
                    val_probs = torch.sigmoid(val_logits).squeeze().cpu().numpy()

                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.hp['patience']:
                    print(f"      Early stopping kích hoạt tại Epoch {epoch}")
                    break

            # Nạp lại trọng số tốt nhất của fold
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_num_va, X_text_va)
                final_val_probs = torch.sigmoid(val_logits).squeeze().cpu().numpy()

            oof_predictions[val_idx] = final_val_probs
            oof_targets[val_idx] = self.labels[val_idx]

            fold_auroc = roc_auc_score(self.labels[val_idx], final_val_probs)
            fold_auprc = average_precision_score(self.labels[val_idx], final_val_probs)
            print(f"      Fold {fold} - Val Loss: {best_val_loss:.4f} | AUROC: {fold_auroc:.4f} | AUPRC: {fold_auprc:.4f}")
            fold_results.append({'fold': fold, 'loss': best_val_loss, 'auroc': fold_auroc, 'auprc': fold_auprc})

        # Tổng hợp toàn bộ Out-Of-Fold (OOF) Metrics
        overall_auroc = roc_auc_score(oof_targets, oof_predictions)
        overall_auprc = average_precision_score(oof_targets, oof_predictions)
        auroc_ci_low, auroc_ci_high = bootstrap_metric_ci(oof_targets, oof_predictions, roc_auc_score)
        auprc_ci_low, auprc_ci_high = bootstrap_metric_ci(oof_targets, oof_predictions, average_precision_score)

        brier = brier_score_loss(oof_targets, oof_predictions)
        ece, calib_bins = calculate_ece(oof_predictions, oof_targets)

        # Ngưỡng tối ưu hóa theo F2-score (y tế ưu tiên giảm bỏ sót False Negatives)
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f2, best_thresh = 0.0, 0.5
        for t in thresholds:
            f2 = fbeta_score(oof_targets, (oof_predictions >= t).astype(int), beta=2.0, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_thresh = t

        optimal_preds = (oof_predictions >= best_thresh).astype(int)
        sensitivity = recall_score(oof_targets, optimal_preds, zero_division=0)
        precision = precision_score(oof_targets, optimal_preds, zero_division=0)
        cm = confusion_matrix(oof_targets, optimal_preds)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

        evaluation_report = {
            'auroc': float(overall_auroc),
            'auroc_ci': [auroc_ci_low, auroc_ci_high],
            'auprc': float(overall_auprc),
            'auprc_ci': [auprc_ci_low, auprc_ci_high],
            'brier_score': float(brier),
            'ece': float(ece),
            'calib_bins': calib_bins,
            'optimal_threshold': float(best_thresh),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f2_score': float(best_f2),
            'confusion_matrix': cm.tolist(),
            'fold_results': fold_results,
            'oof_predictions': oof_predictions,
            'oof_targets': oof_targets,
            'hyperparameters': self.hp
        }

        print("\n" + "=" * 70)
        print(" TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ 5-FOLD CV CHUẨN MỰC QUỐC TẾ")
        print("=" * 70)
        print(f"  • AUROC                      : {overall_auroc:.4f} [95% CI: {auroc_ci_low:.4f} - {auroc_ci_high:.4f}]")
        print(f"  • AUPRC                      : {overall_auprc:.4f} [95% CI: {auprc_ci_low:.4f} - {auprc_ci_high:.4f}]")
        print(f"  • Brier Score (Hiệu chuẩn)   : {brier:.4f}")
        print(f"  • ECE (Calibration Error)    : {ece:.4f}")
        print(f"  • Ngưỡng tối ưu lâm sàng     : {best_thresh:.3f}")
        print(f"  • Độ nhạy (Sensitivity/Recall): {sensitivity * 100:.2f}%")
        print(f"  • Độ đặc hiệu (Specificity)  : {specificity * 100:.2f}%")
        print("=" * 70)

        return evaluation_report

if __name__ == "__main__":
    loader = RealMIMICDataLoader()
    data = loader.load_and_preprocess()
    trainer = PhDEarlyWarningTrainer(data)
    results = trainer.run_stratified_kfold(n_splits=5)
    
    # Lưu toàn bộ báo cáo đánh giá và model
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phd_academic_results.pt")
    torch.save(results, save_path)
    print(f"[✓] Đã lưu báo cáo học thuật hoàn chỉnh tại: {save_path}")
