import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

OPEN_DATASETS = {
    "mimic_iv_demo": {
        "name": "MIMIC-IV Demo (v2.2 - Open Access)",
        "base_url": "https://physionet.org/files/mimic-iv-demo/2.2/",
        "folder": "data/mimic_iv_demo"
    },
    "eicu_demo": {
        "name": "eICU-CRD Demo (v2.0.1 - Open Access)",
        "base_url": "https://physionet.org/files/eicu-crd-demo/2.0.1/",
        "folder": "data/eicu_demo"
    }
}

def scan_files(base_url, current_url):
    urls = []
    try:
        r = requests.get(current_url, timeout=20)
        if r.status_code != 200:
            return urls
        soup = BeautifulSoup(r.text, 'html.parser')
        for a in soup.find_all('a'):
            href = a.get('href')
            if not href or href in ['../', './', '/'] or href.startswith('?'):
                continue
            full_url = urljoin(current_url, href)
            if not full_url.startswith(base_url):
                continue
            if full_url.endswith('/'):
                urls.extend(scan_files(base_url, full_url))
            else:
                urls.append(full_url)
    except Exception as e:
        print(f"[!] Lỗi quét {current_url}: {e}")
    return urls

def download_file(url, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"[-] Đã tồn tại: {os.path.basename(out_path)}")
        return True

    temp_path = out_path + ".part"
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(temp_path, 'wb') as f, tqdm(
                total=total,
                unit='B',
                unit_scale=True,
                desc=os.path.basename(out_path),
                leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=256*1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        os.replace(temp_path, out_path)
        return True
    except Exception as e:
        print(f"[!] Lỗi tải {url}: {e}")
        return False

def main():
    workspace = os.path.dirname(os.path.abspath(__file__))
    print("=" * 65)
    print("  TIẾN HÀNH TẢI BỘ DỮ LIỆU Y TẾ OPEN-ACCESS CHUẨN PHYSIONET")
    print("=" * 65)

    for ds_key, ds in OPEN_DATASETS.items():
        print(f"\n[*] Đang xử lý: {ds['name']}")
        target_dir = os.path.join(workspace, ds['folder'])
        print(f"[*] Thư mục lưu: {target_dir}")
        print(f"[*] Đang quét danh sách tệp tin...")

        files = scan_files(ds['base_url'], ds['base_url'])
        print(f"[+] Tìm thấy {len(files)} tệp tin.")

        base_len = len(ds['base_url'])
        success = 0
        for idx, f_url in enumerate(files, 1):
            rel_path = f_url[base_len:].lstrip('/')
            dest = os.path.join(target_dir, rel_path)
            print(f"[{idx}/{len(files)}] Tải: {rel_path}")
            if download_file(f_url, dest):
                success += 1

        print(f"[✓] Hoàn thành tải {ds['name']}: {success}/{len(files)} tệp tin.")

    print("\n" + "=" * 65)
    print("[🎉] TẤT CẢ DỮ LIỆU ĐÃ ĐƯỢC TẢI THÀNH CÔNG VỀ THƯ MỤC CỤC BỘ!")
    print("=" * 65)

if __name__ == "__main__":
    main()
