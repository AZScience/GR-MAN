import os
import sys
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

DATASETS = {
    "mimiciv": {
        "name": "MIMIC-IV (v3.1)",
        "base_url": "https://physionet.org/files/mimiciv/3.1/",
        "default_dir": "mimiciv_data",
        "description": "Bệnh án điện tử ICU: Dữ liệu bảng, xét nghiệm, dấu hiệu sinh tồn, chẩn đoán ICD"
    },
    "mimic_cxr": {
        "name": "MIMIC-CXR-JPG (v2.1.0)",
        "base_url": "https://physionet.org/files/mimic-cxr-jpg/2.1.0/",
        "default_dir": "mimic_cxr_data",
        "description": "Ảnh X-quang lồng ngực & Báo cáo chẩn đoán hình ảnh (Radiology Reports)"
    },
    "eicu": {
        "name": "eICU-CRD (v2.0)",
        "base_url": "https://physionet.org/files/eicu-crd/2.0/",
        "default_dir": "eicu_data",
        "description": "Dữ liệu hồi sức tích cực đa trung tâm (hơn 200 bệnh viện Hoa Kỳ)"
    }
}

class PhysioNetDownloader:
    def __init__(self, username, password, log_callback=None, progress_callback=None):
        self.username = username
        self.password = password
        self.log_callback = log_callback or print
        self.progress_callback = progress_callback
        self.session = requests.Session()
        self.session.auth = (self.username, self.password)
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PhysioNet-Downloader/2.0"
        })
        self.is_cancelled = False

    def log(self, message):
        self.log_callback(message)

    def cancel(self):
        self.is_cancelled = True
        self.log("[*] Đã yêu cầu dừng quá trình tải...")

    def check_access(self, base_url):
        self.log(f"[*] Đang kiểm tra xác thực quyền truy cập tại: {base_url}")
        try:
            res = self.session.get(base_url, timeout=20)
            if res.status_code == 200:
                self.log("[+] Xác thực THÀNH CÔNG! Tài khoản hợp lệ và có quyền truy cập.")
                return True, "Thành công"
            elif res.status_code == 401:
                msg = "Lỗi 401: Sai Username hoặc Password PhysioNet."
                self.log(f"[!] {msg}")
                return False, msg
            elif res.status_code == 403:
                msg = "Lỗi 403: Tài khoản chưa được cấp quyền Credentialed hoặc chưa ký DUA cho bộ dữ liệu này."
                self.log(f"[!] {msg}")
                return False, msg
            else:
                msg = f"Mã HTTP {res.status_code} khi truy cập {base_url}"
                self.log(f"[!] {msg}")
                return False, msg
        except Exception as e:
            msg = f"Lỗi kết nối mạng: {str(e)}"
            self.log(f"[!] {msg}")
            return False, msg

    def scan_files(self, url, base_url, extensions_filter=None):
        """Quét đệ quy tìm danh sách tệp tin cần tải"""
        if self.is_cancelled:
            return []
            
        file_urls = []
        try:
            res = self.session.get(url, timeout=25)
            if res.status_code != 200:
                self.log(f"[!] Không thể quét thư mục {url} (HTTP {res.status_code})")
                return file_urls

            soup = BeautifulSoup(res.text, 'html.parser')
            for link in soup.find_all('a'):
                if self.is_cancelled:
                    break
                href = link.get('href')
                if not href or href in ['../', './', '/'] or href.startswith('?'):
                    continue

                full_url = urljoin(url, href)
                if not full_url.startswith(base_url):
                    continue

                if full_url.endswith('/'):
                    # Thư mục con
                    file_urls.extend(self.scan_files(full_url, base_url, extensions_filter))
                else:
                    if extensions_filter:
                        if any(full_url.lower().endswith(ext.lower()) for ext in extensions_filter):
                            file_urls.append(full_url)
                    else:
                        file_urls.append(full_url)
        except Exception as e:
            self.log(f"[!] Lỗi khi quét {url}: {e}")

        return file_urls

    def download_single_file(self, file_url, output_path):
        if self.is_cancelled:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temp_file = output_path + ".part"
        initial_byte = os.path.getsize(temp_file) if os.path.exists(temp_file) else 0

        if os.path.exists(output_path):
            self.log(f"[-] Đã tồn tại: {os.path.basename(output_path)} (Bỏ qua)")
            if self.progress_callback:
                self.progress_callback(1.0, 1.0, os.path.basename(output_path), 0)
            return True

        headers = {}
        if initial_byte > 0:
            headers['Range'] = f'bytes={initial_byte}-'

        try:
            with self.session.get(file_url, headers=headers, stream=True, timeout=40) as r:
                if r.status_code == 416: # Range invalid, có thể file đã đủ
                    initial_byte = 0
                    headers.pop('Range', None)
                    r = self.session.get(file_url, stream=True, timeout=40)

                r.raise_for_status()
                content_len = r.headers.get('content-length')
                total_size = (int(content_len) + initial_byte) if content_len else None

                mode = 'ab' if initial_byte > 0 else 'wb'
                downloaded = initial_byte
                start_time = time.time()
                last_update = start_time

                with open(temp_file, mode) as f:
                    for chunk in r.iter_content(chunk_size=512 * 1024): # 512KB
                        if self.is_cancelled:
                            self.log("[*] Đã hủy tải giữa chừng.")
                            return False
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if self.progress_callback and (now - last_update >= 0.2):
                                speed = downloaded / max(1e-6, (now - start_time))
                                self.progress_callback(downloaded, total_size or downloaded, os.path.basename(output_path), speed)
                                last_update = now

            os.replace(temp_file, output_path)
            self.log(f"[✓] Đã tải xong: {os.path.basename(output_path)}")
            return True
        except Exception as e:
            self.log(f"[!] Lỗi tải {file_url}: {e}")
            return False
