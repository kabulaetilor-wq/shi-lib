"""Service HTTP de diagnostic SHI — bibliothèque standard seulement (plus NumPy pour le modèle).

    SHI_API_KEY=une-longue-clé python server.py          (port : variable PORT, 8000 par défaut)

POST /diagnose   en-tête X-API-Key   corps JSON {"description": "...", "logs": "..."}
GET  /health
"""
import hmac
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from shi.diagnosis.service import MAX_DESCRIPTION, MAX_LOGS, diagnose_text
from shi.models.classifier import load_default_classifier

MAX_BODY = MAX_LOGS + MAX_DESCRIPTION + 10_000
RATE_PER_MIN = int(os.environ.get("SHI_RATE_PER_MIN", "60"))
_hits = []
_lock = threading.Lock()


def rate_ok() -> bool:
    now = time.time()
    with _lock:
        while _hits and _hits[0] < now - 60:
            _hits.pop(0)
        if len(_hits) >= RATE_PER_MIN:
            return False
        _hits.append(now)
        return True


def make_handler(api_key: str, classifier):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SHI"
        timeout = 15

        def log_message(self, fmt, *args):  # pas de contenu des requêtes dans les journaux
            print("%s %s" % (self.command, self.path.split("?")[0]))

        def _send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self._send(200, {"status": "ok", "model": classifier is not None})
            self._send(404, {"error": "introuvable"})

        def do_POST(self):
            if self.path != "/diagnose":
                return self._send(404, {"error": "introuvable"})
            if not hmac.compare_digest(self.headers.get("X-API-Key", "").encode(), api_key.encode()):
                return self._send(401, {"error": "authentification requise"})
            if not rate_ok():
                return self._send(429, {"error": "trop de requêtes, réessayez dans une minute"})
            if "application/json" not in self.headers.get("Content-Type", ""):
                return self._send(415, {"error": "JSON requis"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "requête invalide"})
            if n <= 0 or n > MAX_BODY:
                return self._send(413 if n > MAX_BODY else 400, {"error": "taille de requête invalide"})
            try:
                data = json.loads(self.rfile.read(n).decode("utf-8"))
                desc, logs = data.get("description"), data.get("logs", "")
                if not isinstance(desc, str) or len(desc.strip()) < 3 or not isinstance(logs, str):
                    raise ValueError
            except (ValueError, UnicodeDecodeError, AttributeError):
                return self._send(400, {"error": "description (texte, 3 caractères minimum) requise"})
            try:
                self._send(200, diagnose_text(desc, logs, classifier))
            except Exception as exc:  # jamais de détail interne au client
                print("erreur de diagnostic :", type(exc).__name__)
                self._send(500, {"error": "erreur interne"})

    return Handler


def main():
    key = os.environ.get("SHI_API_KEY", "")
    if len(key) < 16:
        raise SystemExit("Définir SHI_API_KEY (16 caractères minimum).")
    clf = load_default_classifier()
    if clf is None:
        raise SystemExit("Aucun modèle dans artifacts/classifier : lancer « python -m shi.cli train-classifier ».")
    port = int(os.environ.get("PORT", "8000"))
    print("SHI en écoute sur le port", port)
    ThreadingHTTPServer(("0.0.0.0", port), make_handler(key, clf)).serve_forever()


if __name__ == "__main__":
    main()
