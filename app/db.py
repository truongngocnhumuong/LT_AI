"""
SQLite database để lưu lịch sử dự đoán.
"""
import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "history.db")


def get_conn():
    """Tạo connection tới SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Khởi tạo bảng nếu chưa có."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            text        TEXT NOT NULL,
            model       TEXT NOT NULL,
            label       TEXT NOT NULL,
            confidence  REAL,
            source      TEXT DEFAULT 'single'
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(text, model, label, confidence, source="single"):
    """Lưu một bản ghi dự đoán."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions (timestamp, text, model, label, confidence, source)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        text[:1000],   # giới hạn độ dài
        model,
        label,
        float(confidence) if confidence is not None else None,
        source,
    ))
    conn.commit()
    conn.close()


def get_history(limit=200, label_filter=None, model_filter=None):
    """Lấy danh sách lịch sử."""
    conn = get_conn()
    cur = conn.cursor()
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    if label_filter and label_filter != "all":
        query += " AND label = ?"
        params.append(label_filter)
    if model_filter and model_filter != "all":
        query += " AND model = ?"
        params.append(model_filter)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_stats():
    """Thống kê tổng quan."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM predictions")
    total = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE label='spam'")
    n_spam = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE label='ham'")
    n_ham = cur.fetchone()[0]

    cur.execute("""
        SELECT model, COUNT(*) AS cnt
        FROM predictions GROUP BY model ORDER BY cnt DESC
    """)
    by_model = [dict(r) for r in cur.fetchall()]

    conn.close()
    return {
        "total": total,
        "spam": n_spam,
        "ham": n_ham,
        "by_model": by_model,
    }


def clear_history():
    """Xoá toàn bộ lịch sử."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
