# Steering hand angle detector
import cv2
import math
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import socket
import time # <--- Aggiungilo all'inizio del file con gli altri import

# --- Aggiungi queste assieme alle altre variabili dello sterzo ---
calibrazione_in_corso = False
inizio_calibrazione_tempo = 0.0
durata_calibrazione = 2.0  # Durata della calibrazione in secondi
somma_angoli_calibrazione = 0.0
conteggio_angoli_calibrazione = 0


# Configurazione: IP locale e una porta libera a scelta

UDP_IP = "127.0.0.1"

UDP_PORT = 4242

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


angolo_continuo = 0.0
ultimo_angolo_filtrato = None

def invia_comando(messaggio: str):

    # Trasforma la stringa in byte e la invia

    sock.sendto(messaggio.encode(), (UDP_IP, UDP_PORT))

    print(f"Inviato: {messaggio}")


# --- FUNZIONI HELPER ---

def lerp(prev, new, alpha):
    """
    Filtro esponenziale:
    alpha alto = più stabile, meno reattivo
    """
    return prev * alpha + new * (1 - alpha)


def angle_diff(a, b):
    """
    Differenza corretta tra due angoli.
    Evita il salto 179 -> -179.
    """
    return (a - b + 180) % 360 - 180


def compute_palm_center(hand_landmarks, width, height):
    """
    Costruisce un punto stabile del palmo usando più landmark.
    Landmark usati:
    0  = wrist
    5  = base indice
    9  = base medio
    13 = base anulare
    17 = base mignolo
    """
    weights = {
        0: 0.10,
        5: 0.20,
        9: 0.30,
        13: 0.20,
        17: 0.20
    }

    x = 0
    y = 0

    for idx, w in weights.items():
        x += hand_landmarks[idx].x * width * w
        y += hand_landmarks[idx].y * height * w

    return [x, y]


# --- ONE EURO FILTER ---

class LowPassFilter:
    def __init__(self, alpha, initval=0.0):
        self.alpha = alpha
        self.s = initval
        self.initialized = False

    def filter(self, value, alpha=None):
        if alpha is not None:
            self.alpha = alpha

        if not self.initialized:
            self.s = value
            self.initialized = True
            return value

        self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def last_value(self):
        return self.s


class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.03, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)

        self.x_filter = LowPassFilter(self.compute_alpha(self.mincutoff))
        self.dx_filter = LowPassFilter(self.compute_alpha(self.dcutoff))
        self.last_time = None

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, value, timestamp=None):
        if self.last_time is not None and timestamp is not None:
            dt = timestamp - self.last_time
            if dt > 1e-8:
                self.freq = 1.0 / dt

        self.last_time = timestamp

        if self.x_filter.initialized:
            dx = (value - self.x_filter.last_value()) * self.freq
        else:
            dx = 0.0

        edx = self.dx_filter.filter(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(edx)

        return self.x_filter.filter(value, self.compute_alpha(cutoff))


# --- 1. DOWNLOAD DEL MODELLO ---

modello_path = "hand_landmarker.task"

if not os.path.exists(modello_path):
    print("Sto scaricando il file dell'IA per le mani...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, modello_path)


# --- 2. CONFIGURAZIONE MEDIAPIPE ---

base_options = python.BaseOptions(model_asset_path=modello_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(options)


# --- 3. CONFIGURAZIONE WEBCAM ---

LARGHEZZA, ALTEZZA = 800, 600

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Errore: webcam non trovata.")
    exit()



# --- VARIABILI STERZO ---

angolo_filtrato = 0.0
angolo_continuo = 0.0
ultimo_angolo_filtrato = None

offset_angolo_zero = 0.0
zero_impostato = False

mano_sx_filtrata = None
mano_dx_filtrata = None

mano_sx_prev = None
mano_dx_prev = None

filtro_punti = 0.80
deadzone_angolo = 6
limite_angolo = 180.0

one_euro_steering = OneEuroFilter(
    freq=30.0,
    mincutoff=1.0,
    beta=0.03,
    dcutoff=1.0
)

timestamp_ms = 0
running = True

print("Sistema avviato.")
print("Premi Q per uscire.")
print("Premi R per ricalibrare lo zero.")


# --- LOOP PRINCIPALE ---

while running:

    successo, frame = cap.read()

    if not successo:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # --- DETECTION MANI ---
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_rgb
    )

    timestamp_ms += 33
    risultati = detector.detect_for_video(mp_image, timestamp_ms)

    mani_trovate = []

    if risultati.hand_landmarks:
        for i, hand_landmarks in enumerate(risultati.hand_landmarks):
            palm_center = compute_palm_center(hand_landmarks, LARGHEZZA, ALTEZZA)

            if risultati.handedness and i < len(risultati.handedness):
                label_model = risultati.handedness[i][0].category_name
            else:
                label_model = "Unknown"

            mani_trovate.append({
                "palm_center": palm_center,
                "label_model": label_model,
                "hand_landmarks": hand_landmarks
            })

            for lm in hand_landmarks:
                cx = int(lm.x * frame.shape[1])
                cy = int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 3, (0, 180, 0), -1)

            cx_palm = int((palm_center[0] / LARGHEZZA) * frame.shape[1])
            cy_palm = int((palm_center[1] / ALTEZZA) * frame.shape[0])
            cv2.circle(frame, (cx_palm, cy_palm), 10, (0, 255, 255), -1)

            if label_model == "Left":
                label_display = "Right"
            elif label_model == "Right":
                label_display = "Left"
            else:
                label_display = label_model

            cv2.putText(
                frame,
                label_display,
                (cx_palm + 10, cy_palm - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

    mano_sx_raw = None
    mano_dx_raw = None


    if len(mani_trovate) == 2:

        punti = [m["palm_center"] for m in mani_trovate]

        # Prima inizializzazione
        if mano_sx_prev is None:
            punti.sort(key=lambda p: p[0])
            mano_sx_raw = punti[0]
            mano_dx_raw = punti[1]

        else:
            # Associazione per distanza (tracking)
            d0_sx = (punti[0][0] - mano_sx_prev[0])**2 + (punti[0][1] - mano_sx_prev[1])**2
            d1_sx = (punti[1][0] - mano_sx_prev[0])**2 + (punti[1][1] - mano_sx_prev[1])**2

            if d0_sx < d1_sx:
                mano_sx_raw = punti[0]
                mano_dx_raw = punti[1]
            else:
                mano_sx_raw = punti[1]
                mano_dx_raw = punti[0]

    # aggiorna memoria
    mano_sx_prev = mano_sx_raw
    mano_dx_prev = mano_dx_raw

    if mano_sx_raw is not None and mano_dx_raw is not None:
        if mano_sx_filtrata is None:
            mano_sx_filtrata = [mano_sx_raw[0], mano_sx_raw[1]]
            mano_dx_filtrata = [mano_dx_raw[0], mano_dx_raw[1]]
        else:
            mano_sx_filtrata[0] = lerp(mano_sx_filtrata[0], mano_sx_raw[0], filtro_punti)
            mano_sx_filtrata[1] = lerp(mano_sx_filtrata[1], mano_sx_raw[1], filtro_punti)

            mano_dx_filtrata[0] = lerp(mano_dx_filtrata[0], mano_dx_raw[0], filtro_punti)
            mano_dx_filtrata[1] = lerp(mano_dx_filtrata[1], mano_dx_raw[1], filtro_punti)

        m_sx = mano_sx_filtrata
        m_dx = mano_dx_filtrata

        p1_webcam = (
            int((m_sx[0] / LARGHEZZA) * frame.shape[1]),
            int((m_sx[1] / ALTEZZA) * frame.shape[0])
        )

        p2_webcam = (
            int((m_dx[0] / LARGHEZZA) * frame.shape[1]),
            int((m_dx[1] / ALTEZZA) * frame.shape[0])
        )

        cv2.line(frame, p1_webcam, p2_webcam, (255, 0, 0), 3)

        dx = m_dx[0] - m_sx[0]
        dy = m_dx[1] - m_sx[1]

        angolo_grezzo = math.degrees(math.atan2(dy, dx))

        tempo_corrente = cv2.getTickCount() / cv2.getTickFrequency()

        angolo_filtrato = one_euro_steering.filter(
            angolo_grezzo,
            tempo_corrente
        )

        if ultimo_angolo_filtrato is None:
            ultimo_angolo_filtrato = angolo_filtrato

        delta = angle_diff(angolo_filtrato, ultimo_angolo_filtrato)
        angolo_continuo += delta
        ultimo_angolo_filtrato = angolo_filtrato

        # --- 1. GESTIONE DELLA CALIBRAZIONE ---
        if not zero_impostato and not calibrazione_in_corso:
            # Inizializza i dati per la calibrazione
            calibrazione_in_corso = True
            inizio_calibrazione_tempo = time.time()
            somma_angoli_calibrazione = 0.0
            conteggio_angoli_calibrazione = 0

        if calibrazione_in_corso:
            # Calcola quanto tempo è passato
            tempo_trascorso = time.time() - inizio_calibrazione_tempo
            progresso = min(tempo_trascorso / durata_calibrazione, 1.0)

            # Accumula l'angolo per fare una media precisa
            somma_angoli_calibrazione += angolo_continuo
            conteggio_angoli_calibrazione += 1

            # --- UI: TESTO E BARRA DI CARICAMENTO ---
            cv2.putText(
                frame, 
                "Calibrazione in corso... Tieni dritto!", 
                (150, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (0, 165, 255), 2
            )
            
            # Disegna la barra di avanzamento
            barra_x, barra_y, barra_w, barra_h = 150, 100, 500, 25
            # Sfondo scuro
            cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (50, 50, 50), -1)
            # Riempimento verde in base al progresso
            cv2.rectangle(frame, (barra_x, barra_y), (barra_x + int(barra_w * progresso), barra_y + barra_h), (0, 255, 0), -1)
            # Bordo bianco
            cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (255, 255, 255), 2)

            # Controlla se i 2 secondi sono finiti
            if tempo_trascorso >= durata_calibrazione:
                calibrazione_in_corso = False
                zero_impostato = True
                # Lo zero ora è la media perfetta degli angoli registrati in questi 2 secondi
                offset_angolo_zero = somma_angoli_calibrazione / max(conteggio_angoli_calibrazione, 1)
                print(f"Calibrazione completata! Zero impostato a: {offset_angolo_zero:.2f}")

        else:
            angolo_sterzata = angolo_continuo - offset_angolo_zero

            if abs(angolo_sterzata) < deadzone_angolo:
                angolo_mostrato = 0.0
            else:
                angolo_mostrato = angolo_sterzata


                    # Clamp a ±180°
            angolo_mostrato = max(-limite_angolo, min(angolo_mostrato, limite_angolo))

            # Conversione [-1, +1]
            sterzo_output = angolo_mostrato / limite_angolo
            sterzo_output = max(-1.0, min(sterzo_output, 1.0))
            invia_comando(str(sterzo_output))
            cv2.putText(
                frame,
                f"Sterzo: {angolo_mostrato:.1f} gradi",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Convertito: {sterzo_output:.1f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2
            )

            cv2.putText(
                frame,
                f"Continuo: {angolo_sterzata:.1f}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    else:
        cv2.putText(
            frame,
            "Mostra due mani",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3
        )

    cv2.imshow("Sterzo con le mani", frame)

    tasto = cv2.waitKey(1) & 0xFF

    if tasto == ord("q"):
        running = False

    if tasto == ord("r"):
        zero_impostato = False
        calibrazione_in_corso = False
        print("Richiesta ricalibrazione... tieni le mani dritte.")


# --- CHIUSURA ---

cap.release()
cv2.destroyAllWindows()