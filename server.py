from flask import Flask, request, jsonify
from flask_cors import CORS
import random, string
from tensorflow.keras.models import load_model
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.dqn.policies import MlpPolicy
from diamond import DiamondMinesEnv

app = Flask(__name__)
CORS(app)

model = load_model("diamond_lstm_model.h5")

games = {}

def generate_id(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

@app.route("/start", methods=["POST"])
def start_game():
    data = request.get_json()
    game_id = generate_id()
    client_seed = data.get("client_seed", "default-seed")
    bet_amount = float(data.get("bet_amount", 10.0))
    balance = float(data.get("balance", 100.0))
    
    mines = random.sample(range(25), 3)
    games[game_id] = {
        "client_seed": client_seed,
        "mines": mines,
        "clicked": [],
        "bet_amount": bet_amount,
        "balance": balance - bet_amount,
        "multiplier": 1.0
    }
    return jsonify({
        "game_id": game_id,
        "client_seed": client_seed,
        "mines": [],  # don't send actual mines!
        "balance": games[game_id]["balance"]
    })

@app.route("/ai_suggest", methods=["POST"])
def ai_suggest_tile():
    data = request.get_json()
    game_id = data["game_id"]
    game = games.get(game_id)

    if not game:
        return jsonify({"error": "Game not found"}), 404

    clicked = game["clicked"]

    # --- Enhanced Observation Vector ---
    # Mark clicked tiles as 1, unclicked as 0
    state = np.zeros(25, dtype=np.float32)
    for i in clicked:
        state[i] = 1.0

    # Optional: Add more features later (like previous rewards, step number, etc.)

    # --- Normalize Input (optional but good practice) ---
    input_array = state.reshape((1, 1, 25))  # Shape for LSTM: (batch, time, features)

    # --- Predict Confidence Scores for Each Tile ---
    predictions = model.predict(input_array)[0]

    # --- Ignore Already Clicked Tiles ---
    unclicked_indices = [i for i in range(25) if i not in clicked]

    if not unclicked_indices:
        return jsonify({"error": "No unclicked tiles remaining"}), 400

    # --- Improved Suggestion Logic ---
    # Sort predictions to choose top-N safe-looking tiles
    sorted_tiles = sorted(
        unclicked_indices,
        key=lambda i: predictions[i],
        reverse=True
    )

    # Choose best candidate (can later add randomness/exploration)
    suggestion = sorted_tiles[0]
    confidence = predictions[suggestion]

    return jsonify({
        "suggested_tile": suggestion,
        "confidence": round(float(confidence), 5),
        "top_3_suggestions": sorted_tiles[:3],
        "all_scores": {i: round(float(predictions[i]), 5) for i in sorted_tiles[:5]}
    })


# @app.route("/rl_suggest", methods=["POST"])
# def rl_suggest_tile():
#     data = request.get_json()
#     game_id = data["game_id"]
#     game = games.get(game_id)

#     if not game:
#         return jsonify({"error": "Game not found"}), 404

#     clicked = game["clicked"]
#     obs = [1 if i in clicked else 0 for i in range(25)]

#     action, _ = rl_model.predict(obs)
#     return jsonify({"suggested_tile": int(action)})


@app.route("/click", methods=["POST"])
def click_tile():
    data = request.get_json()
    game_id = data["game_id"]
    index = int(data["index"])

    game = games.get(game_id)
    if not game or index in game["clicked"]:
        return jsonify({"status": "invalid"})

    game["clicked"].append(index)

    if index in game["mines"]:
        return jsonify({
            "status": "mine",
            "balance": game["balance"],
            "multiplier": 0
        })

    game["multiplier"] = round(game["multiplier"] * 1.3, 2)  # example: grow 30% per safe tile
    return jsonify({
        "status": "safe",
        "multiplier": game["multiplier"]
    })

@app.route("/cashout", methods=["POST"])
def cashout():
    data = request.get_json()
    game_id = data["game_id"]
    game = games.get(game_id)

    if not game:
        return jsonify({"error": "Game not found"}), 404

    winnings = round(game["bet_amount"] * game["multiplier"], 2)
    game["balance"] += winnings
    final_balance = game["balance"]

    del games[game_id]

    return jsonify({
        "status": "cashed_out",
        "winnings": winnings,
        "balance": final_balance
    })

if __name__ == "__main__":
    app.run(debug=True)
