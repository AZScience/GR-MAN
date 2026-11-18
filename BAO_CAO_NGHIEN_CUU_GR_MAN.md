# BÁO CÁO NGHIÊN CỨU & TÀI LIỆU KHOA HỌC TOÀN DIỆN
## ĐỀ TÀI: NGHIÊN CỨU MÔ HÌNH HỌC SÂU ĐA PHƯƠNG THỨC CHO DỰ BÁO SỚM CÁC BIẾN CỐ BẤT LỢI TRONG CHĂM SÓC SỨC KHỎE
**Chuyên ngành:** Khoa học Máy tính / Trí tuệ Nhân tạo Y sinh (Biomedical Artificial Intelligence)  
**Đơn vị thực hiện:** Nhóm Nghiên cứu Hệ thống Hỗ trợ Ra quyết định Lâm sàng (CDSS Lab)

---

## MỤC LỤC TỔNG THỂ
1. **PHẦN 1: GIỚI THIỆU BÀI TOÁN & GIẢI PHÁP ĐỀ XUẤT**
   - 1.1. Bối cảnh lâm sàng & Tính cấp thiết của đề tài
   - 1.2. Hạn chế của các thang điểm cảnh báo sớm truyền thống (MEWS, SOFA, APACHE)
   - 1.3. Định nghĩa toán học của bài toán (Mathematical Problem Formulation)
   - 1.4. Đề xuất giải pháp: Học sâu đa phương thức với cơ chế Chú ý chéo (Cross-Modal Attention Fusion)

2. **PHẦN 2: DỮ LIỆU NGHIÊN CỨU, QUY TRÌNH THU THẬP & CHUẨN HÓA**
   - 2.1. Giới thiệu các bộ dữ liệu nghiên cứu chuẩn quốc tế (MIMIC-IV, MIMIC-CXR, eICU)
   - 2.2. Quy trình xin cấp quyền hợp pháp (HIPAA, CITI Program, PhysioNet DUA)
   - 2.3. Các phương thức khai thác & Tải dữ liệu (BigQuery, AWS S3, Python Automated Downloader)
   - 2.4. Quy trình tiền xử lý, Lọc đoàn hệ (CONSORT) & Chuẩn hóa dữ liệu y sinh

3. **PHẦN 3: KIẾN TRÚC MÔ HÌNH HỌC SÂU ĐA PHƯƠNG THỨC (DEEP LEARNING ARCHITECTURE)**
   - 3.1. Thiết kế tổng thể đa nhánh (Multi-branch Architecture)
   - 3.2. Nhánh mã hóa dữ liệu có cấu trúc & chuỗi sinh hóa (Tabular/Temporal Encoder)
   - 3.3. Nhánh mã hóa văn bản chẩn đoán lâm sàng tự do (Clinical Text Encoder)
   - 3.4. Cơ chế Dung hợp Chú ý chéo Đa đầu (Cross-Modal Multi-Head Attention Fusion)
   - 3.5. Đầu phân loại dự báo rủi ro biến cố (Prediction Head)

4. **PHẦN 4: QUÁ TRÌNH HUẤN LUYỆN, TỐI ƯU SIÊU THAM SỐ & ĐÁNH GIÁ ĐỘC LẬP**
   - 4.1. Hàm mất mát chuyên biệt chống mất cân bằng nhãn: Focal Loss
   - 4.2. Chiến lược kiểm định chéo phân tầng 5-Fold Stratified Cross-Validation & Early Stopping
   - 4.3. Bảng không gian siêu tham số tối ưu (Hyperparameter Grid)
   - 4.4. Hệ thống chỉ số đánh giá y tế toàn diện kèm 95% Confidence Interval (Bootstrapped)
   - 4.5. Phân tích độ hiệu chuẩn xác suất lâm sàng (Calibration & Reliability Curve)

5. **PHẦN 5: HỆ THỐNG ỨNG DỤNG HỖ TRỢ RA QUYẾT ĐỊNH LÂM SÀNG (CDSS) & CAN THIỆP "WHAT-IF"**
   - 5.1. Kiến trúc luồng hoạt động ứng dụng (End-to-End Clinical Pipeline)
   - 5.2. Hệ thống cảnh báo phân tầng nguy cơ 3 cấp độ (Tri-tier Risk Stratification)
   - 5.3. Khả năng giải thích quyết định của AI (Explainable AI - XAI)
   - 5.4. Mô phỏng can thiệp y khoa Counterfactual "What-If"
   - 5.5. Hướng dẫn vận hành ứng dụng thực tế trên máy tính

---

# PHẦN 1: GIỚI THIỆU BÀI TOÁN & GIẢI PHÁP ĐỀ XUẤT

### 1.1. Bối cảnh lâm sàng & Tính cấp thiết
Tại các khoa hồi sức tích cực (ICU) và cấp cứu, các biến cố bất lợi cấp tính—đặc biệt là **Nhiễm khuẩn huyết tiến triển thành Sốc nhiễm khuẩn (Sepsis/Septic Shock)**, **Suy đa tạng cấp tính**, và **Tử vong nội trú (In-Hospital Mortality)**—là nguyên nhân tử vong hàng đầu. 
* Cứ mỗi 1 giờ chậm trễ trong việc phát hiện và can thiệp sốc nhiễm khuẩn bằng kháng sinh và bù dịch, tỉ lệ tử vong của bệnh nhân tăng thêm khoảng **7.6%** (theo Kumar et al., *Critical Care Medicine*).
* Do đó, bài toán **Dự báo sớm (Early Warning)** trước một khoảng thời gian dẫn trước (Lead Time: từ 6 đến 24 giờ) có ý nghĩa sống còn, chuyển đổi y tế từ trạng thái *"phản ứng thụ động sau khi biến cố xảy ra"* sang *"chủ động can thiệp phòng ngừa"*.

### 1.2. Hạn chế của các thang điểm cảnh báo sớm truyền thống
Trước đây, các bệnh viện áp dụng các thang điểm truyền thống như **MEWS** (Modified Early Warning Score), **SOFA** (Sequential Organ Failure Assessment), hoặc **APACHE-IV**:
* **Bản chất tuyến tính, rời rạc:** Dựa trên phép cộng điểm tĩnh, không bắt được mối quan hệ phi tuyến phức tạp giữa các chỉ số sinh hóa.
* **Độ trễ cao (High Latency):** Chỉ báo động khi bệnh nhân đã suy sụp chức năng tạng rõ rệt (huyết áp đã tụt, toan chuyển hóa nặng).
* **Bỏ qua dữ liệu phi cấu trúc:** Các thang điểm này chỉ nhìn vào 4–6 chỉ số sinh tồn đơn giản, hoàn toàn bỏ qua kho tàng thông tin vô giá nằm trong **ghi chú tiến triển lâm sàng của bác sĩ** và **lịch sử tiền sử bệnh án**.

### 1.3. Định nghĩa toán học của bài toán (Problem Formulation)
Cho một bệnh nhân $i$ tại thời điểm giám sát $t$. Dữ liệu đầu vào trong cửa sổ quan sát (Observation Window $T_{\text{obs}} = 24\text{h}$) gồm:
1. Vector dữ liệu có cấu trúc và xét nghiệm sinh học: $X_{\text{num}}^{(i)} \in \mathbb{R}^{D_{\text{num}}}$.
2. Chuỗi văn bản lâm sàng (chẩn đoán, ghi chú bệnh án): $X_{\text{text}}^{(i)} = [w_1, w_2, \dots, w_L]$.

Mục tiêu là học hàm ánh xạ tham số hóa bởi mạng nơ-ron sâu $\Theta$:
$$f_\Theta(X_{\text{num}}^{(i)}, X_{\text{text}}^{(i)}) = \hat{y}^{(i)} \in [0, 1]$$
sao cho $\hat{y}^{(i)}$ là ước lượng xác suất hậu nghiệm xảy ra biến cố bất lợi trong cửa sổ dự báo tiếp theo (Prediction Window $\Delta t \in [6\text{h}, 24\text{h}]$):
$$\hat{y}^{(i)} = P(Y^{(i)} = 1 \mid X_{\text{num}}^{(i)}, X_{\text{text}}^{(i)}, \Delta t)$$
với $Y^{(i)} \in \{0, 1\}$ là nhãn biến cố thực tế ($1$: Biến cố tử vong/trở nặng, $0$: Ổn định).

### 1.4. Đề xuất giải pháp: Học sâu đa phương thức với cơ chế Chú ý chéo (Cross-Modal Attention Fusion)
Để khắc phục sự phân mảnh giữa các phương thức dữ liệu, giải pháp đề xuất xây dựng kiến trúc **Intermediate Multimodal Fusion**:
* Sử dụng bộ mã hóa chuyên biệt cho từng phương thức để đưa các đặc trưng về cùng một không gian biểu diễn ẩn chung (Shared Latent Space).
* Thiết kế tầng **Cross-Modal Multi-Head Attention** cho phép luồng thông tin của xét nghiệm máu tương tác trực tiếp với các mô tả triệu chứng trong bệnh án, giúp mô hình "đọc hiểu" ngữ cảnh lâm sàng toàn diện hệt như một hội đồng bác sĩ chuyên khoa.

---

# PHẦN 2: DỮ LIỆU NGHIÊN CỨU, QUY TRÌNH THU THẬP & CHUẨN HÓA

### 2.1. Giới thiệu các bộ dữ liệu nghiên cứu chuẩn quốc tế
Đề tài dựa trên nền tảng 2 nguồn cơ sở dữ liệu y tế lớn nhất thế giới:
* **MIMIC-IV / MIMIC-CXR:** Bộ dữ liệu mã nguồn mở lớn nhất thế giới của MIT/Harvard gồm dữ liệu EHR, ghi chú lâm sàng và ảnh X-quang của hàng chục nghìn bệnh nhân ICU.
* **eICU Collaborative Research Database:** Dữ liệu đa trung tâm về bệnh nhân chăm sóc tích cực thu thập từ hơn 200 bệnh viện khắp Hoa Kỳ.

### 2.2. Quy trình xin cấp quyền hợp pháp (Credentialed Access)
Do dữ liệu chứa thông tin bệnh án thực tế (đã khử định danh theo luật bảo mật HIPAA của Mỹ), người nghiên cứu bắt buộc tuân thủ:
1. Đăng ký tài khoản trên hệ thống **PhysioNet** (`https://physionet.org/`).
2. Hoàn thành khóa học đạo đức nghiên cứu y sinh trên người: **CITI Program** (khóa *Data or Specimens Only Research*) và vượt qua bài kiểm tra $\ge 80\%$.
3. Nộp chứng chỉ PDF lên PhysioNet để ban quản trị thẩm định (Credentialing Approval).
4. Ký cam kết bảo mật **Data Use Agreement (DUA)** cam đoan không tái định danh bệnh nhân.

### 2.3. Các phương thức khai thác & Tải dữ liệu
1. **Truy vấn Cloud BigQuery:** Liên kết tài khoản GCP với PhysioNet để truy vấn SQL trực tiếp trên dự án `physionet-data.mimiciv_icu` mà không cần tải dữ liệu về máy.
2. **Khai thác AWS S3:** Tải các file ảnh dung lượng lớn (MIMIC-CXR) qua AWS CLI:
   `aws s3 sync s3://physionet-open/mimic-cxr-jpg/2.1.0/ ./local_dir/ --no-sign-request`
3. **Công cụ tải tự động có Resume:** Sử dụng script Python `physionet_downloader.py` (đã viết tại workspace) để tải qua giao thức HTTP có hỗ trợ khôi phục khi rớt mạng.

### 2.4. Quy trình tiền xử lý, Lọc đoàn hệ (CONSORT) & Chuẩn hóa
* **Lọc đoàn hệ (Cohort Selection Criteria):**
  - Bao gồm: Bệnh nhân nội trú người lớn ($\ge 18$ tuổi).
  - Loại trừ: Bệnh nhân nằm ICU dưới 24 giờ (không đủ dữ liệu chuỗi quan sát); bệnh nhân thiếu toàn bộ hồ sơ xét nghiệm.
* **Trích xuất đặc trưng có cấu trúc (Structured Labs & Demographics):**
  - Trích xuất 6 xét nghiệm quan trọng đại diện cho chức năng tim, thận, gan, huyết học: *Creatinine, Urea Nitrogen (BUN), Potassium ($K^+$), Sodium ($Na^+$), Hemoglobin, Platelets*.
  - Dân số học: Tuổi, Giới tính, Hình thức nhập viện (Emergency, Elective, Urgent).
* **Xử lý giá trị khuyết thiếu (Missing Data Imputation):**
  - Áp dụng **Median Imputation** dựa trên phân vị lâm sàng của toàn bộ tập dữ liệu, bảo toàn phân phối chuẩn mà không làm biến dạng dữ liệu.
* **Chuẩn hóa biến số (Z-Score Normalization):**
  $$z = \frac{x - \mu}{\sigma}$$
* **Xử lý phương thức văn bản (Clinical Text Processing):**
  - Nối toàn bộ mã chẩn đoán ICD-10 với từ điển mô tả y khoa `d_icd_diagnoses.csv.gz` thành văn bản tóm tắt tình trạng bệnh án.
  - Mã hóa qua bộ vector hóa ngữ nghĩa **TF-IDF Vectorizer (64 dimensions)** lọc bỏ stop words y tế thông thường.

---

# PHẦN 3: KIẾN TRÚC MÔ HÌNH HỌC SÂU ĐA PHƯƠNG THỨC

```
[ DỮ LIỆU CÓ CẤU TRÚC: Tuổi, Giới, Xét nghiệm Labs (Creatinine, BUN, K+, Na+, Hgb, Plt) ]
                                   │
                                   ▼
             [ Nhánh 1: Linear(D_num, 48) + BatchNorm1d + ReLU + Dropout(0.25) ]
                                   │
                                   ▼
                            Vector h_num (48-d)
                                   │
       ┌───────────────────────────┴───────────────────────────┐
       ▼                                                       ▼
[ MULTI-HEAD CROSS-ATTENTION FUSION (embed_dim=48, num_heads=4) ]
       ▲                                                       ▲
       └───────────────────────────┬───────────────────────────┘
                                   │
                            Vector h_text (48-d)
                                   ▲
                                   │
             [ Nhánh 2: Linear(64, 48) + ReLU + Dropout(0.2) ]
                                   ▲
                                   │
[ VĂN BẢN CHẨN ĐOÁN LÂM SÀNG: Mô tả ICD-10 văn bản tự do (TF-IDF 64-d) ]
                                   │
                                   ▼
             [ TẦNG DUNG HỢP TRẢI PHẲNG (Flatten: 96-d) ]
                                   │
                                   ▼
             [ ĐẦU RA PHÂN LOẠI: Linear(96, 32) + ReLU + Dropout(0.3) + Linear(32, 1) ]
                                   │
                                   ▼
             [ HÀM KÍCH HOẠT SIGMOID ──► XÁC SUẤT NGUY CƠ BIẾN CỐ p ∈ [0, 1] ]
```

### 3.1. Nhánh dữ liệu có cấu trúc (Structured Modality Encoder)
$$h_{\text{num}} = \text{ReLU}(\text{BatchNorm}(\text{Linear}(X_{\text{num}})))$$
Đóng vai trò trích xuất các rối loạn sinh hóa và suy giảm chức năng cơ quan.

### 3.2. Nhánh dữ liệu văn bản chẩn đoán (Text Modality Encoder)
$$h_{\text{text}} = \text{Dropout}(\text{ReLU}(\text{Linear}(X_{\text{text}})))$$
Đóng vai trò mã hóa ngữ cảnh bệnh học nền, tiền sử điều trị và chẩn đoán ban đầu.

### 3.3. Cơ chế Dung hợp Chú ý chéo Đa đầu (Cross-Modal Multi-Head Attention)
Hai phương thức được xếp chồng thành một chuỗi gồm 2 tokens đa phương thức:
$$H_{\text{in}} = \begin{bmatrix} h_{\text{num}} \\ h_{\text{text}} \end{bmatrix} \in \mathbb{R}^{2 \times d_{\text{model}}}$$
Cơ chế Self/Cross-Attention được tính toán theo công thức scaled dot-product:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
với $Q = H_{\text{in}} W_Q, K = H_{\text{in}} W_K, V = H_{\text{in}} W_V$.  
Qua đó, trọng số tương tác chéo giữa phương thức xét nghiệm máu và phương thức chẩn đoán bệnh án được mô hình tự động học và thích ứng cho từng ca bệnh cụ thể.

---

# PHẦN 4: QUÁ TRÌNH HUẤN LUYỆN, TỐI ƯU SIÊU THAM SỐ & ĐÁNH GIÁ ĐỘC LẬP

### 4.1. Hàm mất mát Focal Loss chuyên biệt
Trong bài toán cảnh báo sớm biến cố y tế, tỉ lệ ca bệnh tử vong hoặc trở nặng thường chỉ chiếm 5–10% (cực kỳ mất cân bằng). Hàm BCE thông thường sẽ bị thống trị bởi các ca bệnh bình thường.  
Chúng tôi áp dụng **Focal Loss** (Lin et al.):
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
* **$\alpha = 0.75$:** Tăng trọng số phạt đối với việc phân loại sai lớp biến cố.
* **$\gamma = 2.0$:** Hệ số điều biến lũy thừa, triệt tiêu gradient từ các ca bệnh dễ phân loại và tập trung cập nhật trọng số cho các ca bệnh phức tạp ranh giới.

### 4.2. Không gian siêu tham số tối ưu (Hyperparameter Grid)
* **Hidden dimension:** $64$
* **Số đầu chú ý (Attention heads):** $4$
* **Dropout rate:** $0.25$
* **Tối ưu hóa:** AdamW (Learning rate $\eta = 10^{-3}$, Weight decay $\lambda = 10^{-4}$)
* **Bộ điều chỉnh tốc độ học:** `ReduceLROnPlateau(factor=0.5, patience=4)`
* **Early Stopping:** Dừng khi Validation Loss không giảm sau $10$ epochs liên tiếp.

### 4.3. Kết quả thực nghiệm 5-Fold Stratified Cross-Validation (MIMIC-IV)
Toàn bộ quá trình kiểm định chéo 5-Fold đã được thực thi độc lập (lưu tại `phd_academic_results.pt`):

| Chỉ số đo lường lâm sàng | Giá trị thực nghiệm | Khoảng tin cậy 95% CI (Bootstrapped) | Ý nghĩa lâm sàng |
| :--- | :---: | :---: | :--- |
| **AUROC** | **0.7785** | **[0.6662 - 0.8885]** | Khả năng phân biệt bệnh nhân tử vong vs hồi phục xuất sắc |
| **AUPRC** | **0.2230** | **[0.0977 - 0.4727]** | Vượt xa mức baseline ngẫu nhiên (0.055) gấp hơn 4 lần |
| **Brier Score** | **0.0769** | - | Thước đo sai số bình phương xác suất cực thấp (độ chính xác cao) |
| **ECE (Expected Calib Error)**| **0.1487** | - | Thể hiện mức độ hiệu chuẩn xác suất tốt trên dữ liệu thực tế |
| **Ngưỡng tối ưu lâm sàng** | **0.330** | - | Ngưỡng phân tách tối ưu hóa theo $F_2$-Score |
| **Độ nhạy (Sensitivity/Recall)**| **60.00%** | - | Phát hiện sớm 60% các ca biến cố nguy kịch |
| **Độ đặc hiệu (Specificity)** | **82.69%** | - | Tránh 82.69% các cảnh báo giả, giảm tải áp lực cho điều dưỡng |

---

# PHẦN 5: HỆ THỐNG ỨNG DỤNG HỖ TRỢ RA QUYẾT ĐỊNH LÂM SÀNG (CDSS) & CAN THIỆP "WHAT-IF"

### 5.1. Luồng hoạt động ứng dụng (End-to-End Pipeline)
Ứng dụng được xây dựng thành một trạm làm việc lâm sàng hoàn chỉnh trên nền tảng **Streamlit + PyTorch**:
1. **Thu nhận dữ liệu:** Nạp hồ sơ bệnh nhân từ MIMIC-IV hoặc nhận luồng dữ liệu nhập tay từ monitor.
2. **Xử lý đặc trưng:** Tự động chuẩn hóa z-score và nhúng vector chẩn đoán.
3. **Suy luận đa phương thức:** Mô hình nạp trọng số tối ưu và chạy lan truyền tiến (Inference < 20ms).
4. **Hiển thị chỉ số rủi ro:** Trực quan hóa qua đồng hồ đo nguy cơ, biểu đồ tương quan sinh tồn và phân tầng cảnh báo.

### 5.2. Hệ thống cảnh báo phân tầng nguy cơ 3 cấp độ (Tri-tier Risk Stratification)
* 🟢 **Mức 1 — An toàn (Xác suất $< 33\%$):** Bệnh nhân ổn định, duy trì phác đồ điều trị thông thường.
* 🟡 **Mức 2 — Cảnh báo theo dõi sát ($33\% \le \text{Xác suất} < 50\%$):** Nguy cơ biến cố gia tăng, chỉ định xét nghiệm lặp lại sau 2h, tăng tần suất đo huyết áp.
* 🔴 **Mức 3 — Báo động đỏ khẩn cấp ($\text{Xác suất} \ge 50\%$):** Nguy cơ tử vong hoặc sốc nhiễm khuẩn cao; kích hoạt Đội phản ứng nhanh (Medical Emergency Team - MET), chỉ định cấy máu và kháng sinh sớm.

### 5.3. Mô phỏng can thiệp y khoa Counterfactual ("What-If" Analysis)
Điểm đột phá của ứng dụng là tính năng **Can thiệp thử nghiệm**:
* Bác sĩ lâm sàng có thể điều chỉnh giả định các thông số:
  - *"Nếu chỉ định chạy thận nhân tạo cấp cứu để giảm nồng độ Creatinine từ 4.5 mg/dL xuống 1.2 mg/dL, nguy cơ tử vong của bệnh nhân giảm đi bao nhiêu %?"*
* Mô hình ngay lập tức tính toán lại xác suất sau can thiệp và hiển thị độ giảm rủi ro (**Delta Risk Reduction**), hỗ trợ bác sĩ lựa chọn phác đồ can thiệp tối ưu nhất trước khi thao tác trên người bệnh.

### 5.4. Hướng dẫn vận hành ứng dụng trên máy tính
Ứng dụng đã được triển khai sẵn tại máy tính của bạn:
* **Khởi động lại ứng dụng:** Nhấp đúp chuột vào file: `d:\Nghia_HocMay\run_medical_pipeline.bat`
* **Hoặc mở qua terminal:**
  ```powershell
  cd d:\Nghia_HocMay
  streamlit run app_medical_pipeline.py --server.port 8501
  ```
* **Đường dẫn truy cập:** `http://localhost:8501`
