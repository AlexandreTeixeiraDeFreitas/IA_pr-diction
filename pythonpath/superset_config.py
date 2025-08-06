# superset_config.py

import os

FEATURE_FLAGS = {
    "DASHBOARD_RBAC": True,
    "EMBEDDED_SUPERSET": True,
    "EMBEDDABLE_CHARTS": True 
}

TESTING = True
SESSION_COOKIE_SAMESITE = None
ENABLE_PUBLIC_REGISTRATION = True
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "LimitedViewer"
AUTH_USER_REGISTRATION_EMAIL_MANDATORY = False
SECURITY_SEND_REGISTER_EMAIL = False
# SECURITY_REGISTERABLE = False  # Ne sert à rien si ENABLE_PUBLIC_REGISTRATION est True

# Config mail fictive pour débloquer l'inscription
MAIL_SERVER = 'localhost'
MAIL_PORT = 25
MAIL_USE_TLS = False
MAIL_USE_SSL = False
MAIL_USERNAME = ''
MAIL_PASSWORD = ''
MAIL_DEFAULT_SENDER = 'noreply_superset@gmail.com'

RECAPTCHA_PUBLIC_KEY = ""
RECAPTCHA_PRIVATE_KEY = ""
WTF_CSRF_ENABLED = False

# Autoriser l'accès public aux dashboards
ENABLE_PUBLIC_DASHBOARD_ACCESS = False

# Autoriser les iframes depuis localhost:8081 (ton frontend Vue)
TALISMAN_ENABLED = True
TALISMAN_CONFIG = {
    "content_security_policy": {
        "default-src": [
            "'self'",
            "http://localhost:8081",
            "http://localhost:8089",
            "'unsafe-inline'",        # autorise les styles inline
            "data:",                  # pour certaines polices et images
        ],
        "style-src": [
            "'self'",
            "'unsafe-inline'",       # nécessaire pour le CSS Superset
            "http://localhost:8089",
        ],
        "script-src": [
            "'self'",
            "'unsafe-inline'",       # parfois requis pour le JS Superset
            "http://localhost:8089",
        ],
        "frame-ancestors": [
            "'self'",
            "http://localhost:8081",  # autorise l'iframe depuis Vue.js
        ],
    },
    "frame_options": None,
    "force_https": False,
}

CORS_OPTIONS = {
    'supports_credentials': True,
    'allow_headers': ['*'],
    'resources': ['*'],
    'origins': [
        'http://localhost:8081',  # ton front Vue.js
    ],
}

EMBEDDED_SUPERSET = {
    "JWT_SECRET": os.getenv("SUPERSET_JWT_SECRET", "your-jwt-secret"),
    "JWT_ALG": "HS256",
    "JWT_EXP_SECONDS": 300,
    "JWT_AUDIENCE": "superset",
}

# (si usage du SDK officiel avec /security/guest_token)
GUEST_ROLE_NAME = "Gamma"
GUEST_TOKEN_JWT_SECRET = os.getenv("SUPERSET_JWT_SECRET", "your-jwt-secret")
GUEST_TOKEN_JWT_AUDIENCE = "superset"