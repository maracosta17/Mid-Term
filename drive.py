import argparse
import base64
import os
from io import BytesIO

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image

import cv2
from keras import layers, models, initializers

# ---------------- Server setup ----------------
sio = socketio.Server()
app = Flask(__name__) if (Flask := __import__("flask").Flask) else None  # lazy import to avoid reorder issues
model = None

# ---------------- Arquitectura (equivalente al JSON Keras 1.1.1) ----------------
# JSON original (resumen):
# Input: (80,160,3)
# Conv2D(24, 5x5, relu) -> MaxPool(2x2, s=2)
# Conv2D(36, 5x5, relu) -> MaxPool(2x2, s=2)
# Conv2D(48, 5x5, relu) -> MaxPool(2x2, s=2)
# Conv2D(64, 3x3, relu)
# Conv2D(64, 3x3, relu)
# Flatten -> Dropout(0.1)
# Dense(1024, relu) -> Dropout(0.1)
# Dense(100, relu) -> Dense(50, relu) -> Dense(10, relu) -> Dense(1, linear)

def build_legacy_model():
    Glorot = initializers.GlorotUniform()
    Normal = initializers.RandomNormal()

    m = models.Sequential(name="legacy_keras1_model")
    m.add(layers.InputLayer(input_shape=(80, 160, 3), name="input_1"))

    m.add(layers.Conv2D(
        filters=24, kernel_size=(5,5), strides=(1,1),
        padding="valid", activation="relu",
        kernel_initializer=Glorot, bias_initializer="zeros",
        name="convolution2d_1"
    ))
    m.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid", name="maxpooling2d_1"))

    m.add(layers.Conv2D(
        filters=36, kernel_size=(5,5), strides=(1,1),
        padding="valid", activation="relu",
        kernel_initializer=Glorot, bias_initializer="zeros",
        name="convolution2d_2"
    ))
    m.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid", name="maxpooling2d_2"))

    m.add(layers.Conv2D(
        filters=48, kernel_size=(5,5), strides=(1,1),
        padding="valid", activation="relu",
        kernel_initializer=Glorot, bias_initializer="zeros",
        name="convolution2d_3"
    ))
    m.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="valid", name="maxpooling2d_3"))

    m.add(layers.Conv2D(
        filters=64, kernel_size=(3,3), strides=(1,1),
        padding="valid", activation="relu",
        kernel_initializer=Glorot, bias_initializer="zeros",
        name="convolution2d_4"
    ))
    m.add(layers.Conv2D(
        filters=64, kernel_size=(3,3), strides=(1,1),
        padding="valid", activation="relu",
        kernel_initializer=Glorot, bias_initializer="zeros",
        name="convolution2d_5"
    ))

    m.add(layers.Flatten(name="flatten_1"))
    m.add(layers.Dropout(rate=0.1, name="dropout_1"))

    m.add(layers.Dense(
        units=1024, activation="relu",
        kernel_initializer=Glorot, bias_initializer="zeros",
        name="dense_1"
    ))
    m.add(layers.Dropout(rate=0.1, name="dropout_2"))

    m.add(layers.Dense(100, activation="relu", kernel_initializer=Glorot, bias_initializer="zeros", name="dense_2"))
    m.add(layers.Dense(50, activation="relu", kernel_initializer=Glorot, bias_initializer="zeros", name="dense_3"))
    m.add(layers.Dense(10, activation="relu", kernel_initializer=Glorot, bias_initializer="zeros", name="dense_4"))
    m.add(layers.Dense(1, activation="linear", kernel_initializer=Normal, bias_initializer="zeros", name="dense_5"))
    return m

# ---------------- Preproceso: resize a 80x160 como el JSON ----------------
def preprocess(img_rgb):
    # Sin recorte según tu código anterior: solo resize y normalizar
    # PIL entrega RGB -> np.uint8
    # Redimensionar a (160, 80) (width, height) con OpenCV y luego normalizar
    img = cv2.resize(img_rgb, (160, 80), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img

# ---------------- Socket Handlers ----------------
@sio.on('telemetry')
def telemetry(sid, data):
    if not data:
        sio.emit('manual', data={}, skip_sid=True)
        return

    speed = float(data.get("speed", 0.0))
    img_b64 = data.get("image")
    if img_b64 is None:
        return

    # Decodificar imagen
    image = Image.open(BytesIO(base64.b64decode(img_b64)))
    image_array = np.asarray(image)  # RGB
    X = preprocess(image_array)[None, :, :, :]

    # Predicción
    steering_angle = float(model.predict(X, batch_size=1))

    # Acelera fijo para confirmar movimiento; ajusta 0.20–0.40 según necesites
    throttle = 0.25
    print(f"steer={steering_angle:.4f} throttle={throttle:.2f} speed={speed:.1f}")

    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect", sid)
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data={'steering_angle': str(steering_angle), 'throttle': str(throttle)}, skip_sid=True)

# ---------------- Main ----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving (Keras 1.x model port)')
    parser.add_argument('model', type=str,
                        help='Ruta a model.h5 (pesos) o model.json (si pasas .json, buscará el .h5 con el mismo nombre).')
    args = parser.parse_args()

    model_arg = os.path.abspath(args.model)
    if not os.path.isfile(model_arg):
        print(f"[ERROR] No se encontró: {model_arg}")
        raise SystemExit(1)

    # Determinar archivo de pesos
    if model_arg.lower().endswith('.json'):
        weights_file = model_arg[:-5] + '.h5'
    else:
        weights_file = model_arg

    if not os.path.isfile(weights_file):
        print(f"[ERROR] No se encontraron pesos .h5: {weights_file}")
        raise SystemExit(1)

    # Construir arquitectura equivalente y cargar pesos
    model = build_legacy_model()
    try:
        # Carga directa (mapea por orden); como pusimos mismos nombres, también soporta by_name si lo necesitas
        model.load_weights(weights_file)
        print("[INFO] Pesos cargados correctamente.")
    except Exception as e:
        print(f"[WARN] load_weights directo falló: {e}")
        try:
            model.load_weights(weights_file, by_name=True, skip_mismatch=False)
            print("[INFO] Pesos cargados por nombre.")
        except Exception as e2:
            print(f"[ERROR] No se pudieron cargar los pesos: {e2}")
            raise SystemExit(1)

    # Servidor
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
