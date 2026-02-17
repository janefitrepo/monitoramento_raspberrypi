import cv2
import time
from ultralytics import YOLO
from prometheus_client import Gauge, start_http_server

# Inicia servidor de métricas (porta 8000)
start_http_server(8000)

# Métrica de tempo de inferência
INFERENCE_TIME = Gauge(
    "yolo_inference_time_seconds",
    "Tempo de inferência do YOLO em segundos"
)

# Carrega o modelo YOLO
model = YOLO("yolov8n.pt")
# Abre o vídeo de tráfego
cap = cv2.VideoCapture("videos/transito.mp4")

# Parâmetro de resolução
IMG_SIZE = 640   # testar também com 320

while True:
    ret, frame = cap.read()

    # Reinicia o vídeo quando termina
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Início da inferência
    start = time.time()

    results = model(
        frame,
        imgsz=IMG_SIZE,
        conf=0.4,
        classes=[2]   # Classe 2 = carro
    )

    # Fim da inferência
    end = time.time()

    # Calcula tempo de inferência
    inference_time = end - start

    # Envia métrica ao Prometheus
    INFERENCE_TIME.set(inference_time)

    # Exibe resultado
    annotated = results[0].plot()
    cv2.imshow("YOLO Stream", annotated)

    # Encerra com 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
