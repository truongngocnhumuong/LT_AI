"""
Flask application factory.
"""
import os
from flask import Flask
from .routes import bp as main_bp
from .db import init_db


def create_app():
    """Khởi tạo Flask app."""
    app = Flask(__name__,
                template_folder="templates",
                static_folder="static")

    # Cấu hình
    app.config["SECRET_KEY"] = "spam-classifier-secret-key-change-in-prod"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB cho upload CSV
    app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    # Khởi tạo database lưu lịch sử
    init_db()

    # Đăng ký blueprint
    app.register_blueprint(main_bp)

    return app
