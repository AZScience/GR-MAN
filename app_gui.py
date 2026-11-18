import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from physionet_core import DATASETS, PhysioNetDownloader

class PhysioNetGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PhysioNet Healthcare Dataset Downloader - MIMIC-IV / MIMIC-CXR / eICU")
        self.geometry("780x680")
        self.minsize(700, 600)

        # Biến trạng thái
        self.downloader = None
        self.download_thread = None

        self._apply_style()
        self._create_widgets()

    def _apply_style(self):
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        self.configure(bg="#f4f6f9")
        self.style.configure(".", font=("Segoe UI", 10))
        self.style.configure("TLabel", background="#f4f6f9", foreground="#212529")
        self.style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"), foreground="#0d6efd")
        self.style.configure("TLabelframe", background="#f4f6f9")
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), background="#f4f6f9", foreground="#0d6efd")
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background="#0d6efd", foreground="white")
        self.style.map("Primary.TButton", background=[("active", "#0b5ed7")])
        self.style.configure("Danger.TButton", font=("Segoe UI", 10, "bold"), background="#dc3545", foreground="white")
        self.style.map("Danger.TButton", background=[("active", "#bb2d3b")])

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Tiêu đề
        lbl_title = ttk.Label(main_frame, text="HỆ THỐNG TẢI DỮ LIỆU Y TẾ CHUẨN QUỐC TẾ (PHYSIONET)", style="Header.TLabel")
        lbl_title.pack(anchor="w", pady=(0, 10))

        # Khung 1: Chọn Bộ Dữ Liệu
        frame_ds = ttk.LabelFrame(main_frame, text=" 1. Lựa chọn bộ dữ liệu nghiên cứu ", padding="10")
        frame_ds.pack(fill=tk.X, pady=(0, 10))

        self.var_dataset = tk.StringVar(value="mimiciv")
        for key, val in DATASETS.items():
            rb = ttk.Radiobutton(
                frame_ds, 
                text=f"{val['name']} — {val['description']}", 
                value=key, 
                variable=self.var_dataset,
                command=self._on_dataset_change
            )
            rb.pack(anchor="w", pady=2)

        # Khung 2: Tùy chọn lọc file & Thư mục lưu
        frame_opt = ttk.LabelFrame(main_frame, text=" 2. Cấu hình tệp tin & Thư mục lưu ", padding="10")
        frame_opt.pack(fill=tk.X, pady=(0, 10))

        self.var_filter = tk.BooleanVar(value=True)
        cb_filter = ttk.Checkbutton(
            frame_opt, 
            text="[Khuyên dùng] Chỉ tải các tệp Bảng / Nhãn / Bệnh án (.csv.gz, .csv, .json, .txt) — Bỏ qua ảnh nặng", 
            variable=self.var_filter
        )
        cb_filter.pack(anchor="w", pady=(0, 6))

        frame_dir = ttk.Frame(frame_opt)
        frame_dir.pack(fill=tk.X)
        ttk.Label(frame_dir, text="Thư mục lưu:").pack(side=tk.LEFT, padx=(0, 5))
        self.entry_dir = ttk.Entry(frame_dir)
        self.entry_dir.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        default_save = os.path.join(os.getcwd(), DATASETS["mimiciv"]["default_dir"])
        self.entry_dir.insert(0, default_save)

        btn_browse = ttk.Button(frame_dir, text="Chọn thư mục...", command=self._browse_dir)
        btn_browse.pack(side=tk.LEFT)

        # Khung 3: Tài khoản PhysioNet
        frame_auth = ttk.LabelFrame(main_frame, text=" 3. Thông tin tài khoản PhysioNet (Đã duyệt DUA) ", padding="10")
        frame_auth.pack(fill=tk.X, pady=(0, 10))

        frame_cred = ttk.Frame(frame_auth)
        frame_cred.pack(fill=tk.X)
        
        ttk.Label(frame_cred, text="Username:").grid(row=0, column=0, sticky="w", padx=(0, 5), pady=3)
        self.entry_user = ttk.Entry(frame_cred, width=24)
        self.entry_user.grid(row=0, column=1, sticky="w", padx=(0, 20), pady=3)

        ttk.Label(frame_cred, text="Password:").grid(row=0, column=2, sticky="w", padx=(0, 5), pady=3)
        self.entry_pass = ttk.Entry(frame_cred, width=24, show="*")
        self.entry_pass.grid(row=0, column=3, sticky="w", padx=(0, 10), pady=3)

        btn_test = ttk.Button(frame_cred, text="Kiểm tra kết nối", command=self._test_connection)
        btn_test.grid(row=0, column=4, sticky="w", pady=3)

        # Khung Điều khiển & Tiến trình
        frame_ctrl = ttk.Frame(main_frame)
        frame_ctrl.pack(fill=tk.X, pady=(0, 5))

        self.btn_start = ttk.Button(frame_ctrl, text="BẮT ĐẦU TẢI DỮ LIỆU", style="Primary.TButton", command=self._start_download)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = ttk.Button(frame_ctrl, text="DỪNG TẢI", style="Danger.TButton", command=self._stop_download, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT)

        self.lbl_status = ttk.Label(frame_ctrl, text="Trạng thái: Sẵn sàng", foreground="#6c757d", font=("Segoe UI", 9, "italic"))
        self.lbl_status.pack(side=tk.RIGHT)

        # Tiến trình tải
        self.lbl_file_info = ttk.Label(main_frame, text="Tệp đang tải: Chưa có", font=("Segoe UI", 9))
        self.lbl_file_info.pack(anchor="w", pady=(5, 2))

        self.progress_bar = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))

        # Khung Nhật ký log
        lbl_log = ttk.Label(main_frame, text="Nhật ký tiến trình (Logs):")
        lbl_log.pack(anchor="w")

        self.txt_log = tk.Text(main_frame, height=10, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9), relief=tk.FLAT)
        self.txt_log.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

        scrollbar = ttk.Scrollbar(self.txt_log, command=self.txt_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log['yscrollcommand'] = scrollbar.set

    def _on_dataset_change(self):
        ds_key = self.var_dataset.get()
        new_dir = os.path.join(os.getcwd(), DATASETS[ds_key]["default_dir"])
        self.entry_dir.delete(0, tk.END)
        self.entry_dir.insert(0, new_dir)

    def _browse_dir(self):
        chosen = filedialog.askdirectory(initialdir=self.entry_dir.get(), title="Chọn thư mục lưu dữ liệu")
        if chosen:
            self.entry_dir.delete(0, tk.END)
            self.entry_dir.insert(0, chosen)

    def log(self, text):
        self.after(0, lambda: self._append_log(text))

    def _append_log(self, text):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)

    def _update_progress(self, current, total, filename, speed_bytes):
        percent = (current / total) * 100 if total else 0
        speed_mb = speed_bytes / (1024 * 1024)
        curr_mb = current / (1024 * 1024)
        tot_mb = (total / (1024 * 1024)) if total else 0

        info_text = f"Đang tải: {filename} ({curr_mb:.1f}MB / {tot_mb:.1f}MB) - {percent:.1f}% - Tốc độ: {speed_mb:.2f} MB/s"
        
        self.after(0, lambda: [
            self.progress_bar.configure(value=percent),
            self.lbl_file_info.configure(text=info_text)
        ])

    def _test_connection(self):
        user = self.entry_user.get().strip()
        pwd = self.entry_pass.get().strip()
        if not user or not pwd:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập Username và Password PhysioNet trước khi kiểm tra!")
            return

        ds_key = self.var_dataset.get()
        base_url = DATASETS[ds_key]["base_url"]
        
        def run_test():
            downloader = PhysioNetDownloader(user, pwd, log_callback=self.log)
            ok, msg = downloader.check_access(base_url)
            if ok:
                messagebox.showinfo("Thành công", f"Xác thực PhysioNet thành công!\nTài khoản có quyền truy cập {DATASETS[ds_key]['name']}.")
            else:
                messagebox.showerror("Thất bại", msg)

        threading.Thread(target=run_test, daemon=True).start()

    def _start_download(self):
        user = self.entry_user.get().strip()
        pwd = self.entry_pass.get().strip()
        save_dir = self.entry_dir.get().strip()

        if not user or not pwd:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ Username và Password PhysioNet!")
            return
        if not save_dir:
            messagebox.showwarning("Cảnh báo", "Vui lòng chỉ định thư mục lưu dữ liệu!")
            return

        ds_key = self.var_dataset.get()
        ds_info = DATASETS[ds_key]

        ext_filter = [".csv", ".csv.gz", ".json", ".txt", ".html"] if self.var_filter.get() else None

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.lbl_status.configure(text="Trạng thái: Đang quét và tải...", foreground="#0d6efd")

        def worker():
            self.downloader = PhysioNetDownloader(
                user, pwd, 
                log_callback=self.log, 
                progress_callback=self._update_progress
            )
            
            # 1. Kiểm tra quyền
            ok, msg = self.downloader.check_access(ds_info["base_url"])
            if not ok:
                self.log(f"[!] Hủy bỏ: {msg}")
                self._finish_ui()
                return

            # 2. Quét file
            self.log(f"[*] Đang quét danh sách file trên PhysioNet: {ds_info['base_url']}...")
            files = self.downloader.scan_files(ds_info["base_url"], ds_info["base_url"], ext_filter)
            total = len(files)
            self.log(f"[+] Tìm thấy {total} tệp tin phù hợp.")

            if total == 0:
                self.log("[!] Không tìm thấy tệp tin nào.")
                self._finish_ui()
                return

            # 3. Tải tuần tự
            base_len = len(ds_info["base_url"])
            success_count = 0
            for idx, file_url in enumerate(files, start=1):
                if self.downloader.is_cancelled:
                    break
                rel_path = file_url[base_len:].lstrip('/')
                out_path = os.path.join(save_dir, rel_path)
                self.log(f"[{idx}/{total}] Đang xử lý: {rel_path}")
                
                if self.downloader.download_single_file(file_url, out_path):
                    success_count += 1

            if self.downloader.is_cancelled:
                self.log("[*] Quá trình tải đã bị người dùng tạm dừng.")
            else:
                self.log(f"[🎉] HOÀN TẤT: Đã tải thành công {success_count}/{total} tệp tin về {save_dir}")
                messagebox.showinfo("Thành công", f"Đã hoàn thành quá trình tải {ds_info['name']}!")

            self._finish_ui()

        self.download_thread = threading.Thread(target=worker, daemon=True)
        self.download_thread.start()

    def _stop_download(self):
        if self.downloader:
            self.downloader.cancel()
        self.btn_stop.configure(state=tk.DISABLED)
        self.lbl_status.configure(text="Trạng thái: Đang dừng...", foreground="#dc3545")

    def _finish_ui(self):
        self.after(0, lambda: [
            self.btn_start.configure(state=tk.NORMAL),
            self.btn_stop.configure(state=tk.DISABLED),
            self.lbl_status.configure(text="Trạng thái: Sẵn sàng", foreground="#6c757d"),
            self.progress_bar.configure(value=0),
            self.lbl_file_info.configure(text="Tệp đang tải: Chưa có")
        ])

if __name__ == "__main__":
    app = PhysioNetGUI()
    app.mainloop()
