import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# --- Servidor Socket.IO + Flask ---
sio = socketio.Server(async_mode='eventlet')
app = Flask(__name__)

# Carga del modelo (asegúrate que model.h5 está junto a este archivo)
model = load_model('model.h5')

# Preproceso: redimensionar a (80, 160) y normalizar
def preprocess(img_rgb):
    # Si tu dataset estaba ya en RGB, NO conviertas a BGR.
    # Recorta si quieres quitar cielo/capo (opcional)
    # img_rgb = img_rgb[20:-10, :, :]  # ejemplo de recorte
    img_resized = cv2.resize(img_rgb, (160, 80), interpolation=cv2.INTER_AREA)
    img_float = img_resized.astype(np.float32) / 255.0
    img_float = img_float - 0.5  # centrar
    return img_float

# Control simple del acelerador (constante). Puedes cambiarlo a un PI/PID si quieres.
DEFAULT_THROTTLE = 0.2

@sio.on('telemetry')
def telemetry(sid, data):
    if data is None:
        return

    # El simulador envía 'image' en base64, más steering/throttle actuales, etc.
    img_str = data.get('image', None)
    if img_str is None:
        # sin imagen no podemos predecir
        send_control(0.0, 0.0)
        return

    # Decodificar imagen base64 -> PIL -> np.array (RGB)
    image = Image.open(BytesIO(base64.b64decode(img_str)))
    img_rgb = np.asarray(image)  # esto viene en RGB

    # Preprocesar
    x = preprocess(img_rgb)
    x = np.expand_dims(x, axis=0)  # (1, 80, 160, 3)

    # Predecir dirección
    steering = float(model.predict(x, verbose=0)[0])

    # Acelerador fijo (puedes ajustarlo)
    throttle = DEFAULT_THROTTLE

    # Enviar comando al simulador
    send_control(steering, throttle)

@sio.on('connect')
def connect(sid, environ):
    print('Connected:', sid)
    # enviar un primer control neutro
    send_control(0.0, 0.0)

def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

if __name__ == '__main__':
    # Montar WSGI y escuchar en 4567
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
