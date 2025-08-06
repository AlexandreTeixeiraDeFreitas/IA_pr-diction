from flask import Flask, jsonify
from flask_cors import CORS
import jwt
import datetime
import os

app = Flask(__name__)
CORS(app, origins=["http://localhost:8081"])  # autorise uniquement ton frontend

JWT_SECRET = os.getenv("SUPERSET_JWT_SECRET", "your-jwt-secret")
JWT_AUDIENCE = os.getenv("GUEST_TOKEN_JWT_AUDIENCE", "superset")

@app.route("/guest_token/<dashboard_uuid>")
def guest_token(dashboard_uuid):
    payload = {
        "user": {
            "username": "guest@frontend",
            "first_name": "Guest",
            "last_name": "User"
        },
        "resources": [
            {"type": "dashboard", "id": dashboard_uuid}
        ],
        "rls_rules": [],
        "iat": int(datetime.datetime.utcnow().timestamp()),
        "exp": int((datetime.datetime.utcnow() + datetime.timedelta(minutes=5)).timestamp()),
        "aud": JWT_AUDIENCE,
        "type": "guest"
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return jsonify({"token": token})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)
