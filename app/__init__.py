from flask import Flask
from flask_cors import CORS
from app.config import SECRET_KEY, FLASK_DEBUG


def create_app():
    app = Flask(
        __name__,
        template_folder="../frontend/templates",
        static_folder="../frontend/static",
    )

    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["DEBUG"] = FLASK_DEBUG

    CORS(app)

    from app.services.feedback import init_db
    init_db()

    from app.routes.chat import chat_bp
    from app.routes.health import health_bp
    from app.routes.eval import eval_bp

    app.register_blueprint(chat_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(eval_bp)

    return app