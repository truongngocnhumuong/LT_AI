"""
Entry point của ứng dụng web phân loại Spam.
Chạy: python run.py
"""
from app.app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 60)
    print("  SPAM CLASSIFIER — Học Máy và Ứng Dụng")
    print("=" * 60)
    print("  Mở trình duyệt tại: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5000, debug=True)
