import os
import sys
import argparse
import getpass
from physionet_core import DATASETS, PhysioNetDownloader

def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu PhysioNet y tế (MIMIC-IV, MIMIC-CXR, eICU)")
    parser.add_argument("--dataset", choices=["mimiciv", "mimic_cxr", "eicu"], default="mimiciv",
                        help="Tên bộ dữ liệu (mimiciv, mimic_cxr, eicu)")
    parser.add_argument("--username", "-u", type=str, help="PhysioNet Username")
    parser.add_argument("--password", "-p", type=str, help="PhysioNet Password")
    parser.add_argument("--output", "-o", type=str, help="Thư mục lưu")
    parser.add_argument("--all-files", action="store_true", help="Tải toàn bộ file kể cả ảnh/dữ liệu nặng (mặc định chỉ tải bảng/nhãn)")

    args = parser.parse_args()

    ds_info = DATASETS[args.dataset]
    print("=" * 60)
    print(f"[*] Bộ dữ liệu: {ds_info['name']}")
    print(f"[*] Mô tả: {ds_info['description']}")
    print("=" * 60)

    username = args.username or input("Nhập PhysioNet Username: ").strip()
    password = args.password or getpass.getpass("Nhập PhysioNet Password: ").strip()
    output_dir = args.output or os.path.join(os.getcwd(), ds_info["default_dir"])

    ext_filter = None if args.all_files else [".csv", ".csv.gz", ".json", ".txt", ".html"]
    if ext_filter:
        print("[*] Đang áp dụng bộ lọc an toàn: Chỉ tải dữ liệu bảng/nhãn (csv, csv.gz, json).")

    downloader = PhysioNetDownloader(username, password)
    
    # 1. Kiểm tra xác thực
    ok, msg = downloader.check_access(ds_info["base_url"])
    if not ok:
        print(f"[!] Dừng: {msg}")
        sys.exit(1)

    # 2. Quét file
    print(f"[*] Đang quét danh sách file tại {ds_info['base_url']} ...")
    files = downloader.scan_files(ds_info["base_url"], ds_info["base_url"], ext_filter)
    total = len(files)
    print(f"[+] Tìm thấy tổng cộng {total} tệp tin.")

    if total == 0:
        print("[!] Không có tệp tin nào để tải.")
        return

    # 3. Tiến hành tải
    base_len = len(ds_info["base_url"])
    success = 0
    for idx, f_url in enumerate(files, start=1):
        rel = f_url[base_len:].lstrip('/')
        out_path = os.path.join(output_dir, rel)
        print(f"[{idx}/{total}] Đang xử lý: {rel}")
        if downloader.download_single_file(f_url, out_path):
            success += 1

    print("\n" + "=" * 60)
    print(f"[+] Hoàn thành: Đã tải {success}/{total} tệp tin về {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
