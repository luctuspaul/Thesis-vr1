import qrcode

registration_url = "http://localhost:5000/register"

qr = qrcode.make(registration_url)
qr.save("website/static/registration_qr.png")

print(" QR Code generated: registration_qr.png")
