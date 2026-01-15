from flask import Flask, render_template
import qrcode
import uuid
import os

app = Flask(__name__)

QR_FOLDER = os.path.join("static", "qr")
os.makedirs(QR_FOLDER, exist_ok=True)

# PAGE THAT SHOWS QR
@app.route("/activate")
def activate():
    token = str(uuid.uuid4())

    register_url = f"http://localhost:5000/register/{token}"

    qr_filename = f"{token}.png"
    qr_full_path = os.path.join(QR_FOLDER, qr_filename)

    qr = qrcode.make(register_url)
    qr.save(qr_full_path)

    return render_template(
        "activate.html",
        qr_filename=f"qr/{qr_filename}"
    )

# QR SCAN → AUTO REGISTER
@app.route("/register/<token>")
def register(token):
    print(f" Registered token: {token}")
    return "<h2>Registration Complete</h2>"

if __name__ == "__main__":
    app.run(debug=True)
