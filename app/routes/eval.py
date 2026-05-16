import json
from pathlib import Path
from flask import Blueprint, request, jsonify, render_template
from app.services.feedback import save_feedback, get_stats, get_all_feedback

eval_bp = Blueprint("eval", __name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "evaluation" / "results.json"


@eval_bp.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json()

    if not data:
        return jsonify({"error": "No data provided"}), 400

    required = ["session_id", "question", "answer", "vote"]
    missing  = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    if data["vote"] not in ("up", "down"):
        return jsonify({"error": "vote must be 'up' or 'down'"}), 400

    try:
        save_feedback(
            session_id = data["session_id"],
            question   = data["question"],
            answer     = data["answer"],
            vote       = data["vote"],
        )
        return jsonify({"status": "saved", "vote": data["vote"]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route("/stats", methods=["GET"])
def stats():
    try:
        return jsonify(get_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@eval_bp.route("/dashboard")
def dashboard():
    eval_results = None
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            eval_results = json.load(f)

    feedback_stats = get_stats()
    feedback_log   = get_all_feedback()

    return render_template(
        "dashboard.html",
        eval_results   = eval_results,
        feedback_stats = feedback_stats,
        feedback_log   = feedback_log,
    )