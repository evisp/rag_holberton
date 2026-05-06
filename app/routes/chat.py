from flask import Blueprint, render_template, request, jsonify
from app.services.rag_chain import ask
from app.services.memory import get_memory

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/")
def index():
    return render_template("index.html")


@chat_bp.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()

    if not data or not data.get("question", "").strip():
        return jsonify({"error": "No question provided"}), 400

    question   = data["question"].strip()
    session_id = data.get("session_id")

    result = ask(question, session_id=session_id)

    return jsonify({
        "question":   result["question"],
        "answer":     result["answer"],
        "sources":    result["sources"],
        "model":      result["model"],
        "session_id": result["session_id"],
    })


@chat_bp.route("/clear", methods=["POST"])
def clear_session():
    data       = request.get_json()
    session_id = data.get("session_id") if data else None

    if session_id:
        get_memory().clear_session(session_id)

    return jsonify({"status": "cleared"})