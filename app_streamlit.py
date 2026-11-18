import os
import streamlit as st
import time
from physionet_core import DATASETS, PhysioNetDownloader

st.set_page_config(
    page_title="PhysioNet Dataset Downloader",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 PhysioNet Healthcare Dataset Downloader")
st.markdown("Công cụ tải dữ liệu nghiên cứu y sinh: **MIMIC-IV**, **MIMIC-CXR**, và **eICU** hỗ trợ tải tiếp tục khi đứt mạng (Resume capability).")

# Sidebar cấu hình
st.sidebar.header("🔑 Xác thực tài khoản PhysioNet")
username = st.sidebar.text_input("Username PhysioNet", placeholder="vd: nguyen_van_a")
password = st.sidebar.text_input("Password PhysioNet", type="password")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Lưu ý:** Bạn cần hoàn thành chứng chỉ CITI Program và được cấp quyền Credentialed trên PhysioNet trước khi tải.")

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Chọn bộ dữ liệu")
    ds_options = {
        "mimiciv": f"MIMIC-IV (v3.1) - {DATASETS['mimiciv']['description']}",
        "mimic_cxr": f"MIMIC-CXR-JPG (v2.1.0) - {DATASETS['mimic_cxr']['description']}",
        "eicu": f"eICU-CRD (v2.0) - {DATASETS['eicu']['description']}"
    }
    selected_key = st.radio("Bộ dữ liệu cần tải:", list(ds_options.keys()), format_func=lambda k: ds_options[k])
    ds_info = DATASETS[selected_key]

with col2:
    st.subheader("2. Tùy chọn tải & Thư mục lưu")
    filter_mode = st.radio(
        "Chế độ tải tệp tin:",
        [
            "Chỉ tải tệp Bảng / Bệnh án / Nhãn (.csv.gz, .csv, .json, .txt) [Khuyên dùng, nhẹ]",
            "Tải toàn bộ tệp tin (Bao gồm cả ảnh/sóng dung lượng lớn hàng trăm GB)"
        ]
    )
    is_filtered = "Chỉ tải" in filter_mode
    default_dir = os.path.join(os.getcwd(), ds_info["default_dir"])
    save_dir = st.text_input("Thư mục lưu cục bộ:", value=default_dir)

# Kiểm tra kết nối
if st.button("🔍 Kiểm tra quyền truy cập (Test Credentials)"):
    if not username or not password:
        st.warning("Vui lòng điền Username và Password ở cột bên trái.")
    else:
        with st.spinner("Đang kiểm tra với server PhysioNet..."):
            downloader = PhysioNetDownloader(username, password)
            ok, msg = downloader.check_access(ds_info["base_url"])
            if ok:
                st.success(f"✅ Xác thực thành công! Bạn có quyền truy cập {ds_info['name']}.")
            else:
                st.error(f"❌ {msg}")

st.markdown("---")
st.subheader("3. Bắt đầu tải")

if st.button("🚀 BẮT ĐẦU TẢI DỮ LIỆU", type="primary"):
    if not username or not password:
        st.error("Vui lòng nhập Username và Password của PhysioNet trước khi tải.")
    elif not save_dir:
        st.error("Vui lòng nhập thư mục lưu hợp lệ.")
    else:
        status_box = st.empty()
        log_box = st.empty()
        progress_bar = st.progress(0)
        logs = []

        def log_fn(msg):
            logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            log_box.code("\n".join(logs[-15:]), language="bash")

        downloader = PhysioNetDownloader(username, password, log_callback=log_fn)
        
        status_box.info("Đang kiểm tra quyền truy cập...")
        ok, msg = downloader.check_access(ds_info["base_url"])
        if not ok:
            status_box.error(f"Không thể tiếp tục: {msg}")
        else:
            status_box.info(f"Đang quét danh sách file tại {ds_info['base_url']}...")
            ext_filter = [".csv", ".csv.gz", ".json", ".txt", ".html"] if is_filtered else None
            files = downloader.scan_files(ds_info["base_url"], ds_info["base_url"], ext_filter)
            total = len(files)
            
            if total == 0:
                status_box.warning("Không tìm thấy tệp tin nào phù hợp.")
            else:
                status_box.success(f"Tìm thấy {total} tệp tin! Bắt đầu tải...")
                base_len = len(ds_info["base_url"])
                success_count = 0
                
                for idx, f_url in enumerate(files, start=1):
                    rel = f_url[base_len:].lstrip('/')
                    out_path = os.path.join(save_dir, rel)
                    status_box.info(f"[{idx}/{total}] Đang tải: {rel}")
                    
                    if downloader.download_single_file(f_url, out_path):
                        success_count += 1
                    progress_bar.progress(int((idx / total) * 100))
                    
                status_box.success(f"🎉 Hoàn tất! Đã tải thành công {success_count}/{total} tệp tin về {save_dir}")
