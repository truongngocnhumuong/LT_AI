"""
Service load các mô hình đã huấn luyện và thực hiện dự đoán.
Hỗ trợ 7 mô hình:
  - 5 ML cổ điển: KNN, Naive Bayes, Decision Tree, KMeans, SVM
  - 2 Deep Learning: LSTM, DistilBERT
"""
import os
import re
import json
import pickle
import numpy as np

# Đường dẫn thư mục models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# ============================================================
# Tiền xử lý văn bản (phải đồng nhất với notebook training)
# ============================================================
def preprocess(text: str) -> str:
    """Làm sạch text: lowercase, thay số/url, loại ký tự đặc biệt."""
    text = str(text).lower()
    text = re.sub(r"\d+", "NUM", text)
    text = re.sub(r"http\S+|www\S+", "URL", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# Lazy-load models
# ============================================================
_state = {
    "tfidf": None,
    "ml_models": {},      # KNN, NB, DT, KMeans, SVM
    "kmeans_map": None,   # mapping cluster_id -> label
    "lstm": None,
    "lstm_tokenizer": None,
    "lstm_maxlen": 100,
    "distilbert_model": None,
    "distilbert_tokenizer": None,
    "results": {},
    "loaded": False,
}


def _load_pickle(filename):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all():
    """Load tất cả mô hình từ thư mục models/."""
    if _state["loaded"]:
        return

    print("[ml_service] Loading models...")

    # 1. TF-IDF vectorizer
    _state["tfidf"] = _load_pickle("tfidf_vectorizer.pkl")

    # 2. 5 ML models cổ điển
    ml_files = {
        "KNN": "knn.pkl",
        "Naive Bayes": "naive_bayes.pkl",
        "Decision Tree": "decision_tree.pkl",
        "KMeans": "kmeans.pkl",
        "SVM": "svm.pkl",
    }
    for name, fname in ml_files.items():
        m = _load_pickle(fname)
        if m is not None:
            _state["ml_models"][name] = m
            print(f"  [OK] {name}")

    # KMeans cần mapping cluster -> label
    _state["kmeans_map"] = _load_pickle("kmeans_label_map.pkl")

    # 3. LSTM
    lstm_h5 = os.path.join(MODELS_DIR, "lstm.h5")
    lstm_keras = os.path.join(MODELS_DIR, "lstm.keras")
    lstm_path = lstm_keras if os.path.exists(lstm_keras) else lstm_h5
    if os.path.exists(lstm_path):
        try:
            from tensorflow.keras.models import load_model as keras_load
            _state["lstm"] = keras_load(lstm_path)
            _state["lstm_tokenizer"] = _load_pickle("lstm_tokenizer.pkl")
            cfg = _load_pickle("lstm_config.pkl")
            if cfg and "maxlen" in cfg:
                _state["lstm_maxlen"] = cfg["maxlen"]
            print("  [OK] LSTM")
        except Exception as e:
            print(f"  [FAIL] LSTM: {e}")

    # 4. DistilBERT
    bert_dir = os.path.join(MODELS_DIR, "distilbert")
    if os.path.isdir(bert_dir):
        try:
            from transformers import (AutoTokenizer,
                                      AutoModelForSequenceClassification)
            _state["distilbert_tokenizer"] = AutoTokenizer.from_pretrained(bert_dir)
            _state["distilbert_model"] = (
                AutoModelForSequenceClassification.from_pretrained(bert_dir))
            _state["distilbert_model"].eval()
            print("  [OK] DistilBERT")
        except Exception as e:
            print(f"  [FAIL] DistilBERT: {e}")

    # 5. Results JSON
    results_path = os.path.join(MODELS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            _state["results"] = json.load(f)

    _state["loaded"] = True
    print("[ml_service] Done.")


# ============================================================
# Predict
# ============================================================
def list_available_models():
    """Trả về danh sách mô hình hiện có."""
    load_all()
    available = list(_state["ml_models"].keys())
    if _state["lstm"] is not None:
        available.append("LSTM")
    if _state["distilbert_model"] is not None:
        available.append("DistilBERT")
    return available


def _predict_ml(text_clean, model_name):
    """Dự đoán bằng các ML model cổ điển."""
    tfidf = _state["tfidf"]
    model = _state["ml_models"][model_name]
    vec = tfidf.transform([text_clean])

    if model_name == "KMeans":
        cluster_id = int(model.predict(vec)[0])
        # Map cluster -> label theo majority vote đã lưu khi train
        label = _state["kmeans_map"].get(cluster_id, 0)
        # KMeans không có xác suất; dùng khoảng cách làm "confidence"
        distances = model.transform(vec)[0]
        d_to_predicted = distances[cluster_id]
        d_other = distances[1 - cluster_id]
        # Càng gần cluster của mình so với cluster kia thì confidence càng cao
        prob_predicted = d_other / (d_to_predicted + d_other + 1e-9)
        prob_spam = prob_predicted if label == 1 else 1 - prob_predicted
        return int(label), float(prob_spam)

    pred = int(model.predict(vec)[0])

    # Lấy xác suất spam
    if hasattr(model, "predict_proba"):
        prob_spam = float(model.predict_proba(vec)[0][1])
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(vec)[0])
        prob_spam = 1.0 / (1.0 + np.exp(-score))
    else:
        prob_spam = float(pred)

    return pred, prob_spam


def _predict_lstm(text_clean):
    """Dự đoán bằng LSTM."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    tokenizer = _state["lstm_tokenizer"]
    seq = tokenizer.texts_to_sequences([text_clean])
    pad = pad_sequences(seq, maxlen=_state["lstm_maxlen"], padding="post")
    prob_spam = float(_state["lstm"].predict(pad, verbose=0)[0][0])
    pred = 1 if prob_spam >= 0.5 else 0
    return pred, prob_spam


def _predict_distilbert(text_raw):
    """Dự đoán bằng DistilBERT (dùng text gốc, không tiền xử lý)."""
    import torch

    tok = _state["distilbert_tokenizer"]
    model = _state["distilbert_model"]
    enc = tok(text_raw, truncation=True, padding=True,
              max_length=128, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1).numpy()[0]
    prob_spam = float(probs[1])
    pred = int(np.argmax(probs))
    return pred, prob_spam


def predict(text, model_name="SVM"):
    """Hàm dự đoán chính, trả về dict kết quả."""
    load_all()
    text_clean = preprocess(text)

    if model_name == "LSTM":
        if _state["lstm"] is None:
            raise RuntimeError("Mô hình LSTM chưa được huấn luyện/lưu.")
        pred, prob_spam = _predict_lstm(text_clean)
    elif model_name == "DistilBERT":
        if _state["distilbert_model"] is None:
            raise RuntimeError("Mô hình DistilBERT chưa được huấn luyện/lưu.")
        pred, prob_spam = _predict_distilbert(text)
    elif model_name in _state["ml_models"]:
        pred, prob_spam = _predict_ml(text_clean, model_name)
    else:
        raise ValueError(f"Không tìm thấy mô hình: {model_name}")

    return {
        "model": model_name,
        "label": "spam" if pred == 1 else "ham",
        "label_id": pred,
        "prob_spam": prob_spam,
        "prob_ham": 1.0 - prob_spam,
        "text_clean": text_clean,
    }


def predict_batch(texts, model_name="SVM"):
    """Dự đoán hàng loạt cho list các text."""
    return [predict(t, model_name) for t in texts]


def get_model_results():
    """Trả về kết quả đánh giá đã lưu (từ notebook)."""
    load_all()
    return _state["results"]
