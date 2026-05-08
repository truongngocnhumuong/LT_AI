"""
Setup script — chạy 1 lần trước khi mở notebook.
Copy spam.csv từ thư mục cha (Cuối Kỳ/) vào data/ nếu chưa có.

Cách dùng:
    python setup.py
"""
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

target = DATA / "spam.csv"

if target.exists():
    print(f"[OK] {target} đã tồn tại — bỏ qua.")
else:
    # Tìm spam.csv ở các vị trí khả dĩ
    candidates = [
        ROOT.parent / "spam.csv",          # Cuối Kỳ/spam.csv
        ROOT / "spam.csv",                  # spam_classifier_v2/spam.csv
        ROOT.parent.parent / "spam.csv",   # AI/spam.csv
    ]
    for src in candidates:
        if src.exists():
            shutil.copy(src, target)
            print(f"[OK] Đã copy {src}  ->  {target}")
            break
    else:
        print("[!] Không tìm thấy spam.csv ở các vị trí thông thường.")
        print("    Vui lòng copy thủ công vào: " + str(target))
        raise SystemExit(1)

# Tạo các thư mục cần thiết
for d in ["models", "app/uploads"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

print("\n✅ Setup xong. Các bước tiếp theo:")
print("   1. pip install -r requirements.txt")
print("   2. jupyter notebook notebooks/01_train_models.ipynb  (chạy hết)")
print("   3. python run.py")
