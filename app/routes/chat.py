from flask import Blueprint, render_template, request, jsonify
from app.services.rag_chain import ask

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/")
def index():
    return render_template("index.html")


@chat_bp.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()

    if not data or not data.get("question", "").strip():
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()

    result = ask(question)

    return jsonify({
        "question": result["question"],
        "answer":   result["answer"],
        "sources":  result["sources"],
        "model":    result["model"],
    })