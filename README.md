# 🛡️ Spam Classifier — Đồ án Học Máy và Ứng Dụng

Hệ thống phân loại tin nhắn email/SMS thành **Spam** / **Ham** với **7 mô hình**:
**KNN · Naive Bayes · Decision Tree · KMeans · SVM · LSTM · DistilBERT**

Đóng gói thành ứng dụng web Flask hoàn chỉnh: phân loại 1 tin nhắn, so sánh mô hình, phân loại hàng loạt từ CSV, lưu lịch sử dự đoán bằng SQLite.

---

## 👤 Thông tin

| Mục | Nội dung |
|---|---|
| Môn học | Học Máy và Ứng Dụng |
| Sinh viên | Trương Ngọc Như Muống |
| MSSV | 4551190036 |
| GVHD | Lê Quang Hùng |
| Trường | Đại học Quy Nhơn |

---

## 📁 Cấu trúc thư mục

```
spam_classifier_v2/
├── data/
│   └── spam.csv                  # Dataset SMS Spam Collection
├── notebooks/
│   ├── 01_train_models.ipynb     # Notebook training 7 mô hình
│   └── _build_notebook.py        # (helper sinh notebook, có thể xóa)
├── models/                       # Output của notebook (sinh sau khi train)
│   ├── tfidf_vectorizer.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── decision_tree.pkl
│   ├── kmeans.pkl
│   ├── kmeans_label_map.pkl
│   ├── svm.pkl
│   ├── lstm.h5
│   ├── lstm_tokenizer.pkl
│   ├── lstm_config.pkl
│   ├── distilbert/               # Folder chứa DistilBERT đã fine-tune
│   └── results.json              # Kết quả đánh giá
├── app/
│   ├── __init__.py
│   ├── app.py                    # Flask application factory
│   ├── routes.py                 # Routes của 5 trang web + API
│   ├── ml_service.py             # Load model + dự đoán
│   ├── db.py                     # SQLite lưu lịch sử
│   ├── templates/                # 5 file HTML (Jinja2)
│   │   ├── base.html
│   │   ├── index.html            # Trang phân loại 1 tin
│   │   ├── compare.html          # Trang so sánh 7 mô hình
│   │   ├── batch.html            # Trang upload CSV
│   │   ├── history.html          # Trang lịch sử
│   │   └── about.html            # Trang hướng dẫn
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── run.py                        # Entry point — chạy Flask
├── requirements.txt              # Dependencies
└── README.md                     # File này
```

---

## ⚙️ Cài đặt

Yêu cầu: Python 3.9+ (khuyến khích 3.10 hoặc 3.11).

```bash
# 1. (Khuyến khích) Tạo virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 2. Cài thư viện
pip install -r requirements.txt
```

> **Lưu ý:** Phần DistilBERT cần `torch` và `transformers` (~500 MB). Nếu máy yếu, có thể bỏ qua phần này — Flask vẫn chạy với 6 mô hình còn lại.

---

## 🚀 Cách chạy

### Bước 0: Setup (chạy 1 lần)

```bash
# Copy spam.csv vào data/, tạo các thư mục cần thiết
python setup.py
```

### Bước 1: Train mô hình

```bash
jupyter notebook notebooks/01_train_models.ipynb
```

Sau đó chạy tất cả các cell từ trên xuống dưới (menu *Cell → Run All*).

- Phần 5 ML cổ điển: ~30 giây trên CPU.
- Phần LSTM: ~2-5 phút trên CPU.
- Phần DistilBERT: ~40 phút trên CPU, ~5 phút trên GPU. **Khuyến khích chạy trên Google Colab có GPU**, sau đó tải thư mục `models/distilbert/` về.

Khi chạy xong, thư mục `models/` sẽ chứa tất cả file model + `results.json`.

### Bước 2: Chạy ứng dụng web

```bash
python run.py
```

Mở trình duyệt: **http://127.0.0.1:5000**

---

## 🎯 Tính năng web app

| Trang | Đường dẫn | Mô tả |
|---|---|---|
| Phân loại | `/` | Nhập 1 tin nhắn, chọn mô hình, xem kết quả + xác suất |
| So sánh | `/compare` | Bảng + biểu đồ so sánh 7 mô hình + confusion matrix |
| Hàng loạt | `/batch` | Upload CSV (cột `text` / `v2`), phân loại tối đa 1000 dòng |
| Lịch sử | `/history` | Xem tất cả dự đoán đã thực hiện, lọc theo nhãn / mô hình |
| Hướng dẫn | `/about` | Tài liệu mô tả đồ án |

REST API: `POST /api/predict` với JSON `{"text": "...", "model": "SVM"}`.

---

## 🧪 Test nhanh API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"WINNER! You have won a free iPhone!", "model":"SVM"}'
```

---

## 🛠️ Stack công nghệ

- **Backend:** Flask 3, SQLite
- **ML:** scikit-learn (5 mô hình cổ điển), TF-IDF
- **Deep Learning:** TensorFlow/Keras (LSTM), PyTorch + Hugging Face Transformers (DistilBERT)
- **Frontend:** HTML/CSS/JS thuần, Chart.js cho biểu đồ
- **Notebook:** Jupyter
