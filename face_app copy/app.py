import cv2
import os
import pickle
import base64
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import face_recognition
from PIL import Image
import threading
import shutil
import ssl

Image.MAX_IMAGE_PIXELS = None

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)   # Required for cross-origin browser camera requests when deployed online

# =========================
# ENCODINGS
# =========================
known_encodings = []
known_names     = []
encoding_status = {"running": False, "done": False, "total": 0, "error": ""}

def load_encodings():
    global known_encodings, known_names
    pkl = "encodings.pkl"
    print(f"\n🔍 Looking for {pkl} at: {os.path.abspath(pkl)}")
    print(f"   File exists: {os.path.exists(pkl)}")

    if os.path.exists(pkl):
        try:
            with open(pkl, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, tuple):
                known_encodings, known_names = data
            else:
                known_encodings = data.get("encodings", [])
                known_names     = data.get("names", [])
            known_encodings = [np.array(e, dtype=np.float64) for e in known_encodings]
            print(f"✅ Loaded {len(known_names)} encodings. Names: {list(set(known_names))}")
        except Exception as e:
            print(f"❌ Failed to load encodings.pkl: {e}")
            known_encodings, known_names = [], []
    else:
        known_encodings, known_names = [], []
        print("⚠️  encodings.pkl not found — run /rebuild_encodings once")

load_encodings()


def _encode_faces_thread():
    global known_encodings, known_names, encoding_status
    encoding_status.update({"running": True, "done": False, "error": "", "total": 0})

    new_encs, new_names = [], []
    dataset_path = "dataset"

    if not os.path.exists(dataset_path):
        encoding_status.update({"running": False, "error": "dataset/ folder not found"})
        return

    persons = [d for d in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📁 Encoding persons: {persons}")

    for person_name in persons:
        folder = os.path.join(dataset_path, person_name)
        imgs   = [f for f in os.listdir(folder)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  👤 {person_name}: {len(imgs)} images")
        for fname in imgs:
            img_path = os.path.join(folder, fname)
            try:
                image = face_recognition.load_image_file(img_path)
                encs  = face_recognition.face_encodings(image)
                if encs:
                    new_encs.append(encs[0])
                    new_names.append(person_name)
                    print(f"    ✔ encoded {fname}")
                else:
                    print(f"    ⚠ no face in {fname}")
            except Exception as e:
                print(f"    ✗ error {fname}: {e}")

    with open("encodings.pkl", "wb") as f:
        pickle.dump((new_encs, new_names), f)

    known_encodings = [np.array(e, dtype=np.float64) for e in new_encs]
    known_names     = new_names
    encoding_status.update({"running": False, "done": True, "total": len(new_names)})
    print(f"✅ Encoding complete — {len(new_names)} encodings. Names: {list(set(new_names))}")


# =========================
# LOGGING
# =========================
# Throttle: only log the same name once per 5 seconds to avoid log spam
_last_logged = {}
_log_lock    = threading.Lock()

def log_entry(name):
    now = datetime.now()
    with _log_lock:
        last = _last_logged.get(name)
        if last and (now - last).total_seconds() < 5:
            return
        _last_logged[name] = now
    with open("logs.txt", "a") as f:
        f.write(f"{name},{now}\n")


# =========================
# DETECT ENDPOINT  ← core change for online deployment
# Receives a base64 JPEG frame from the browser,
# runs face recognition, returns name + status + face boxes.
# =========================
@app.route("/detect", methods=["POST"])
def detect():
    data       = request.json or {}
    image_data = data.get("image", "")
    if not image_data:
        return jsonify({"name": "Unknown", "status": "ACCESS DENIED", "faces": []})

    # Decode base64 → numpy image
    if "," in image_data:
        image_data = image_data.split(",")[1]
    try:
        img_bytes = base64.b64decode(image_data)
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("imdecode returned None")
    except Exception as e:
        return jsonify({"name": "Error", "status": "ACCESS DENIED", "faces": [], "error": str(e)})

    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Use 'hog' model (CPU-friendly); switch to 'cnn' if you have a GPU
    locs  = face_recognition.face_locations(rgb, model="hog")
    encs  = face_recognition.face_encodings(rgb, locs)

    results   = []
    top_name   = "Unknown"
    top_status = "ACCESS DENIED"

    for (top, right, bottom, left), encode in zip(locs, encs):
        name   = "Unknown"
        status = "ACCESS DENIED"
        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, encode)
            best      = int(np.argmin(distances))
            if distances[best] < 0.50:
                name   = known_names[best]
                status = "ACCESS GRANTED"
                log_entry(name)

        results.append({
            "name":   name,
            "status": status,
            "box":    {"top": top, "right": right, "bottom": bottom, "left": left}
        })
        # Promote first recognised face as the primary result
        if name != "Unknown":
            top_name   = name
            top_status = status

    if not results:
        return jsonify({"name": "No Face", "status": "DETECTING...", "faces": []})

    return jsonify({"name": top_name, "status": top_status, "faces": results})


# =========================
# ROUTES — PAGES
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/logs")
def view_logs():
    return render_template("logs.html")

@app.route("/users")
def users():
    return render_template("users.html")


# =========================
# ROUTES — API
# =========================
@app.route("/logs_json")
def logs_json():
    logs = []
    if os.path.exists("logs.txt"):
        with open("logs.txt", "r") as f:
            logs = [l.strip() for l in f.readlines() if l.strip()]
    return jsonify(logs[-30:])

@app.route("/known_count")
def known_count():
    return jsonify({"count": len(set(known_names)), "names": list(set(known_names))})

@app.route("/users_list")
def users_list():
    if not os.path.exists("dataset"):
        return jsonify([])
    users = []
    for d in os.listdir("dataset"):
        folder = os.path.join("dataset", d)
        if os.path.isdir(folder):
            imgs = len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
            users.append({"name": d, "images": imgs})
    return jsonify(sorted(users, key=lambda x: x["name"]))

@app.route("/delete_user", methods=["POST"])
def delete_user():
    name = request.json.get("name", "").strip()
    if not name:
        return jsonify({"error": "No name"}), 400
    folder = os.path.join("dataset", name)
    if os.path.exists(folder):
        shutil.rmtree(folder)
        global known_encodings, known_names
        indices = [i for i, n in enumerate(known_names) if n != name]
        known_encodings = [known_encodings[i] for i in indices]
        known_names     = [known_names[i] for i in indices]
        with open("encodings.pkl", "wb") as f:
            pickle.dump((known_encodings, known_names), f)
        return jsonify({"success": True})
    return jsonify({"error": "User not found"}), 404

@app.route("/register_face", methods=["POST"])
def register_face():
    data       = request.json
    name       = data.get("name", "").strip()
    image_data = data.get("image", "")
    if not name or not image_data:
        return jsonify({"error": "Invalid data"}), 400
    if "," in image_data:
        image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    folder      = os.path.join("dataset", name)
    os.makedirs(folder, exist_ok=True)
    count    = len([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])
    img_path = os.path.join(folder, f"{count+1}.jpg")
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    return jsonify({"message": "saved", "count": count+1})

@app.route("/finalize_registration", methods=["POST"])
def finalize_registration():
    if encoding_status["running"]:
        return jsonify({"message": "already_running"})
    encoding_status["done"] = False
    threading.Thread(target=_encode_faces_thread, daemon=True).start()
    return jsonify({"message": "started"})

@app.route("/rebuild_encodings")
def rebuild_encodings():
    if encoding_status["running"]:
        return jsonify({"message": "already_running"})
    encoding_status["done"] = False
    threading.Thread(target=_encode_faces_thread, daemon=True).start()
    return jsonify({"message": "started — poll /encoding_status to check progress"})

@app.route("/encoding_status")
def get_encoding_status():
    return jsonify(encoding_status)

@app.route("/clear_logs", methods=["POST"])
def clear_logs():
    if os.path.exists("logs.txt"):
        open("logs.txt", "w").close()
    return jsonify({"success": True})


# =========================
# HTTPS — auto self-signed cert
# =========================
CERT_FILE = "cert.pem"
KEY_FILE  = "key.pem"

def ensure_ssl_cert():
    """Generate a self-signed cert+key if they don't already exist."""
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    try:
        from OpenSSL import crypto
        key = crypto.PKey()
        key.generate_key(crypto.TYPE_RSA, 2048)

        cert = crypto.X509()
        cert.get_subject().CN = "localhost"
        cert.set_serial_number(1)
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)   # 1 year
        cert.set_issuer(cert.get_subject())
        cert.set_pubkey(key)
        cert.sign(key, "sha256")

        with open(CERT_FILE, "wb") as f:
            f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
        with open(KEY_FILE, "wb") as f:
            f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))
        print("✅ Self-signed SSL cert generated (cert.pem / key.pem)")
    except ImportError:
        print("⚠️  pyopenssl not installed — falling back to HTTP (camera may be blocked)")
        print("   Run:  pip install pyopenssl")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    ensure_ssl_cert()

    ssl_ctx = None
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        ssl_ctx = (CERT_FILE, KEY_FILE)
        print("🔒 Starting on https://0.0.0.0:443")
        print("   Open: https://localhost  (accept the browser warning once)")
    else:
        print("🌐 Starting on http://0.0.0.0:5000  (no SSL)")

    port = 443 if ssl_ctx else 5000
    app.run(
        host="0.0.0.0",
        port=port,
        ssl_context=ssl_ctx,
        debug=False,          # debug=True + ssl_context causes issues
        threaded=True,
        use_reloader=False,
    )