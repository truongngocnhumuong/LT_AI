"""
Routes của ứng dụng Flask.
"""
import os
import io
import csv
from flask import (Blueprint, render_template, request, jsonify,
                   redirect, url_for, flash, current_app)
from werkzeug.utils import secure_filename

from . import ml_service
from . import db

bp = Blueprint("main", __name__)


# ------------------------------------------------------------
# Trang chủ — Phân loại 1 tin nhắn
# ------------------------------------------------------------
@bp.route("/", methods=["GET"])
def index():
    available = ml_service.list_available_models()
    return render_template("index.html",
                           models=available,
                           active_page="index")


@bp.route("/api/predict", methods=["POST"])
def api_predict():
    """API JSON: nhận text + model -> trả kết quả."""
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()
    model_name = data.get("model") or "SVM"

    if not text:
        return jsonify({"error": "Vui lòng nhập nội dung tin nhắn."}), 400

    try:
        result = ml_service.predict(text, model_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Lưu lịch sử
    db.save_prediction(
        text=text,
        model=model_name,
        label=result["label"],
        confidence=max(result["prob_spam"], result["prob_ham"]),
        source="single",
    )

    return jsonify(result)


# ------------------------------------------------------------
# Trang So sánh mô hình
# ------------------------------------------------------------
@bp.route("/compare", methods=["GET"])
def compare():
    results = ml_service.get_model_results()
    return render_template("compare.html",
                           results=results,
                           active_page="compare")


@bp.route("/api/results", methods=["GET"])
def api_results():
    return jsonify(ml_service.get_model_results())


# ------------------------------------------------------------
# Trang Phân loại hàng loạt (Batch)
# ------------------------------------------------------------
@bp.route("/batch", methods=["GET", "POST"])
def batch():
    available = ml_service.list_available_models()

    if request.method == "POST":
        # Lấy file upload
        f = request.files.get("file")
        model_name = request.form.get("model", "SVM")
        text_column = request.form.get("text_column", "text")

        if not f or f.filename == "":
            flash("Chưa chọn file CSV.", "error")
            return redirect(url_for("main.batch"))

        filename = secure_filename(f.filename)
        if not filename.lower().endswith(".csv"):
            flash("Chỉ chấp nhận file .csv", "error")
            return redirect(url_for("main.batch"))

        save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        f.save(save_path)

        # Đọc CSV và dự đoán
        rows = []
        try:
            with open(save_path, "r", encoding="utf-8", errors="ignore") as fp:
                reader = csv.DictReader(fp)
                # Tìm cột text linh hoạt
                fieldnames = reader.fieldnames or []
                col = None
                for cand in [text_column, "text", "v2", "message", "content"]:
                    if cand in fieldnames:
                        col = cand
                        break
                if col is None and fieldnames:
                    col = fieldnames[0]

                for row in reader:
                    txt = (row.get(col) or "").strip()
                    if txt:
                        rows.append(txt)
        except Exception as e:
            flash(f"Lỗi đọc CSV: {e}", "error")
            return redirect(url_for("main.batch"))

        if not rows:
            flash("File CSV không có dữ liệu hợp lệ.", "error")
            return redirect(url_for("main.batch"))

        # Giới hạn số dòng để tránh treo
        MAX_ROWS = 1000
        if len(rows) > MAX_ROWS:
            flash(f"File quá lớn, chỉ xử lý {MAX_ROWS} dòng đầu.", "warning")
            rows = rows[:MAX_ROWS]

        # Predict
        results = []
        for txt in rows:
            try:
                r = ml_service.predict(txt, model_name)
                results.append({
                    "text": txt[:200],
                    "label": r["label"],
                    "prob_spam": r["prob_spam"],
                })
                db.save_prediction(
                    text=txt, model=model_name,
                    label=r["label"],
                    confidence=max(r["prob_spam"], 1 - r["prob_spam"]),
                    source="batch",
                )
            except Exception as e:
                results.append({
                    "text": txt[:200],
                    "label": "error",
                    "prob_spam": 0,
                    "error": str(e),
                })

        n_spam = sum(1 for r in results if r["label"] == "spam")
        n_ham = sum(1 for r in results if r["label"] == "ham")

        return render_template("batch.html",
                               models=available,
                               results=results,
                               n_spam=n_spam,
                               n_ham=n_ham,
                               total=len(results),
                               selected_model=model_name,
                               active_page="batch")

    return render_template("batch.html",
                           models=available,
                           results=None,
                           active_page="batch")


# ------------------------------------------------------------
# Trang Lịch sử
# ------------------------------------------------------------
@bp.route("/history", methods=["GET"])
def history():
    label_filter = request.args.get("label", "all")
    model_filter = request.args.get("model", "all")
    rows = db.get_history(limit=300,
                          label_filter=label_filter,
                          model_filter=model_filter)
    stats = db.get_stats()
    available = ml_service.list_available_models()
    return render_template("history.html",
                           rows=rows,
                           stats=stats,
                           models=available,
                           label_filter=label_filter,
                           model_filter=model_filter,
                           active_page="history")


@bp.route("/history/clear", methods=["POST"])
def history_clear():
    db.clear_history()
    flash("Đã xoá toàn bộ lịch sử.", "success")
    return redirect(url_for("main.history"))


# ------------------------------------------------------------
# Trang Hướng dẫn / About
# ------------------------------------------------------------
@bp.route("/about", methods=["GET"])
def about():
    return render_template("about.html", active_page="about")
