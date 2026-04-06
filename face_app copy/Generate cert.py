"""
generate_cert.py — run this once to create cert.pem and key.pem
Usage:  python generate_cert.py

Requires:  pip install pyopenssl
"""

from OpenSSL import crypto
import os

CERT_FILE = "cert.pem"
KEY_FILE  = "key.pem"

def generate():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        print("✅ cert.pem and key.pem already exist — nothing to do.")
        return

    # Generate RSA key
    key = crypto.PKey()
    key.generate_key(crypto.TYPE_RSA, 2048)

    # Generate self-signed certificate
    cert = crypto.X509()
    subj = cert.get_subject()
    subj.C  = "IN"
    subj.ST = "Punjab"
    subj.L  = "Amritsar"
    subj.O  = "Driver Safety Monitoring"
    subj.CN = "localhost"

    cert.set_serial_number(1000)
    cert.gmtime_adj_notBefore(0)
    cert.gmtime_adj_notAfter(365 * 24 * 60 * 60)  # valid for 1 year
    cert.set_issuer(cert.get_subject())
    cert.set_pubkey(key)

    # Add Subject Alternative Names so modern browsers accept it
    san_list = [
        b"DNS:localhost",
        b"IP:127.0.0.1",
        b"IP:0.0.0.0",
    ]
    cert.add_extensions([
        crypto.X509Extension(b"subjectAltName", False, b", ".join(san_list)),
        crypto.X509Extension(b"basicConstraints", True, b"CA:TRUE"),
    ])
    cert.sign(key, "sha256")

    with open(CERT_FILE, "wb") as f:
        f.write(crypto.dump_certificate(crypto.FILETYPE_PEM, cert))
    with open(KEY_FILE, "wb") as f:
        f.write(crypto.dump_privatekey(crypto.FILETYPE_PEM, key))

    print(f"✅ Generated {CERT_FILE} and {KEY_FILE}")
    print("   Start the app:  python app.py")
    print("   Then open:      https://localhost")
    print("   (Click 'Advanced → Proceed' in your browser to accept the self-signed cert)")

if __name__ == "__main__":
    generate()