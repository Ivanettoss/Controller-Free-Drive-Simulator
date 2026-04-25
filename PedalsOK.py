import cv2
import numpy as np

# ============================================================
# PEDAL VISION - VERSIONE AUTO ROI PER NUOVA STAMPA B/N
# ============================================================
# Flusso:
#   1. Avvia il programma.
#   2. Metti il foglio ben visibile, senza piedi sopra.
#   3. Premi C.
#   4. Il programma trova automaticamente FRENO e GAS.
#   5. Calibra automaticamente i riferimenti.
#   6. Usa i pedali.
#
# Comandi:
#   C = auto-rileva ROI + calibra
#   R = reset
#   S = salva screenshot dashboard
#   Q = esci
# ============================================================

# ================= CONFIG CAMERA =================
CAMERA_ID = 0
FRAME_W, FRAME_H = 640, 480

# ================= AUTO ROI =================
# Con la nuova stampa: pedali neri su foglio bianco.
# Il codice cerca prima il foglio chiaro, poi i due gruppi neri più grandi.
# Più alto = il pavimento grigio non viene scambiato per foglio.
# Nel tuo screenshot 185 era troppo basso: prendeva anche le piastrelle.
PAPER_WHITE_THRESHOLD = 210
PAPER_MIN_AREA_RATIO = 0.08

AUTO_BLACK_FIXED_THRESHOLD = 165
AUTO_MIN_PEDAL_AREA_RATIO = 0.006
AUTO_MIN_PEDAL_W = 35
AUTO_MIN_PEDAL_H = 55

# Pad piccolo: evita di includere fughe/piastrelle dentro la ROI.
AUTO_ROI_PAD_X = 12
AUTO_ROI_PAD_Y = 12

# Se dopo la classificazione per forma risultano ancora invertiti,
# metti True. Di solito deve restare False.
FORCE_SWAP_GAS_BRAKE = False

# Morfologia per unire bordo, puntini e linee interne in un unico blob per pedale.
AUTO_CLOSE_KERNEL = 23
AUTO_DILATE_KERNEL = 15

# ================= CALIBRAZIONE =================
CALIBRATION_FRAMES = 25

# Nero stampato su foglio bianco.
# Durante la calibrazione il threshold del nero viene calcolato con Otsu,
# poi viene limitato dentro questo range per evitare soglie assurde.
# Con stampa bianco/nero la soglia non deve salire troppo:
# 170 includeva anche ombre/pavimento, causando pressione fittizia.
BLACK_THRESHOLD_MARGIN = 6
BLACK_THRESHOLD_MIN = 45
BLACK_THRESHOLD_MAX = 145

# Rilevamento automatico della faccia del pedale, escludendo il gambo.
FACE_ROW_SPAN_RATIO = 0.52
FACE_PAD_X, FACE_PAD_Y = 8, 8

# ================= EDGE / DIFF =================
CANNY_LOW, CANNY_HIGH = 50, 140
EDGE_TOLERANCE_RADIUS = 3
MIN_REF_EDGE_PIXELS = 120
MIN_REF_BLACK_PIXELS = 250
INTENSITY_DIFF_THRESHOLD = 42

# Con la nuova stampa, il segnale più robusto è BLACK_MISSING:
# quando il piede copre il pedale, i dettagli neri stampati spariscono.
BLACK_MISSING_WEIGHT = 0.55
EDGE_MISSING_WEIGHT = 0.30
INTENSITY_WEIGHT = 0.15

# Soglie pressione. Da regolare dopo il primo test.
GAS_HARD_THRESHOLD = 0.07
GAS_FULL_PRESSURE_AT = 0.55
BRAKE_HARD_THRESHOLD = 0.07
BRAKE_FULL_PRESSURE_AT = 0.55

SMOOTHING = 0.22

# ================= DASHBOARD =================
PANEL_W, PANEL_H = 220, 145
HEADER_H, FOOTER_H = 22, 22
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ================= DISPLAY =================
# False = mostra solo la schermata principale a colori con pedali, box e barre.
# True  = mostra tutta la griglia debug.
SHOW_DEBUG_DASHBOARD = True

WINDOW_NAME = "Pedal Vision Output"

# 1.0 = 640x480
# 1.5 = più grande
# 2.0 = molto grande
OUTPUT_SCALE = 1.0
# ============================================================
# UTILS BASE
# ============================================================
def clamp_roi(roi):
    x, y, w, h = map(int, roi)
    x = max(0, min(x, FRAME_W - 1))
    y = max(0, min(y, FRAME_H - 1))
    w = max(1, min(w, FRAME_W - x))
    h = max(1, min(h, FRAME_H - y))
    return x, y, w, h


def expand_roi(roi, pad_x=AUTO_ROI_PAD_X, pad_y=AUTO_ROI_PAD_Y):
    x, y, w, h = map(int, roi)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(FRAME_W, x + w + pad_x)
    y2 = min(FRAME_H, y + h + pad_y)
    return clamp_roi((x1, y1, x2 - x1, y2 - y1))


def crop(img, roi):
    x, y, w, h = clamp_roi(roi)
    return img[y:y + h, x:x + w]


def normalize_camera_gray(gray):
    """
    Normalizza la luminosità: il bianco del foglio viene portato vicino a 255.
    Questo aiuta quando la webcam scurisce o cambia esposizione.
    """
    g = gray.astype(np.float32)
    p95 = np.percentile(g, 95)
    if p95 > 20:
        g *= 255.0 / p95
    return np.clip(g, 0, 255).astype(np.uint8)


def preprocess(frame):
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_norm = normalize_camera_gray(gray_raw)

    # CLAHE solo debug: non lo uso per la logica principale.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_norm)

    # Logica principale: immagine normalizzata e leggermente sfocata.
    gray_logic = cv2.GaussianBlur(gray_norm, (3, 3), 0)
    return frame, gray_raw, gray_norm, gray_clahe, gray_logic


def get_edges(gray):
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)


def map_pressure(raw, hard_threshold, full_pressure_at):
    if raw < hard_threshold:
        return 0.0
    return float(np.clip((raw - hard_threshold) / (full_pressure_at - hard_threshold), 0.0, 1.0))


def response_curve(x):
    return float(np.clip(x, 0.0, 1.0) ** 1.18)


def smooth(prev, curr):
    return prev * (1.0 - SMOOTHING) + curr * SMOOTHING


# ============================================================
# AUTO RILEVAMENTO FOGLIO + PEDALI
# ============================================================
def largest_contour_mask(mask, min_area=0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        if cv2.contourArea(c) >= min_area:
            out = np.zeros_like(mask)
            cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
            return out, c

    return None, None


def remove_long_thin_noise(mask):
    """
    Elimina fughe delle piastrelle e linee sottili lunghe.
    Serve prima del merge morfologico, altrimenti una fuga del pavimento
    può attaccarsi al pedale e finire dentro la ROI.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)

    for c in contours:
        area = cv2.contourArea(c)
        if area < 12:
            continue

        x, y, w, h = cv2.boundingRect(c)
        rect_area = max(1, w * h)
        fill = area / rect_area
        long_side = max(w, h)
        short_side = min(w, h)

        # Fughe/piastrelle: lunghe, sottili, basso riempimento.
        if long_side > 75 and fill < 0.045:
            continue

        # Linee quasi monodimensionali molto lunghe.
        if short_side <= 3 and long_side > 45:
            continue

        cv2.drawContours(cleaned, [c], -1, 255, thickness=cv2.FILLED)

    return cleaned


def detect_paper_mask(gray_logic):
    """
    Cerca il foglio chiaro. Se fallisce, ritorna una maschera full-frame.
    """
    paper = np.zeros_like(gray_logic, dtype=np.uint8)
    paper[gray_logic >= PAPER_WHITE_THRESHOLD] = 255

    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    paper = cv2.morphologyEx(paper, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    min_area = int(FRAME_W * FRAME_H * PAPER_MIN_AREA_RATIO)
    paper_mask, paper_contour = largest_contour_mask(paper, min_area=min_area)

    if paper_mask is None:
        paper_mask = np.ones_like(gray_logic, dtype=np.uint8) * 255
        paper_rect = (0, 0, FRAME_W, FRAME_H)
        return paper_mask, paper_rect, False

    paper_rect = cv2.boundingRect(paper_contour)
    return paper_mask, paper_rect, True


def auto_detect_pedal_rois(gray_logic):
    """
    Trova automaticamente le ROI dei due pedali.
    Assegnazione:
        pedale nero più a sinistra = FRENO
        pedale nero più a destra   = GAS
    """
    paper_mask, paper_rect, paper_found = detect_paper_mask(gray_logic)

    # Nero candidato, solo dentro il foglio o dentro la maschera fallback.
    black = np.zeros_like(gray_logic, dtype=np.uint8)
    black[(gray_logic <= AUTO_BLACK_FIXED_THRESHOLD) & (paper_mask > 0)] = 255

    # Pulisce rumore piccolo e soprattutto rimuove fughe diagonali delle piastrelle.
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    black = remove_long_thin_noise(black)

    # Unisce tutti i dettagli interni del singolo pedale in una macro-forma.
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (AUTO_CLOSE_KERNEL, AUTO_CLOSE_KERNEL))
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (AUTO_DILATE_KERNEL, AUTO_DILATE_KERNEL))
    merged = cv2.morphologyEx(black, cv2.MORPH_CLOSE, close_k)
    merged = cv2.dilate(merged, dilate_k, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    frame_area = FRAME_W * FRAME_H
    min_area = frame_area * AUTO_MIN_PEDAL_AREA_RATIO

    px, py, pw, ph = paper_rect
    paper_center_x = px + pw / 2.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(c)
        if w < AUTO_MIN_PEDAL_W or h < AUTO_MIN_PEDAL_H:
            continue

        # Evita componenti enormi tipo ombre o bordi del foglio.
        if w > FRAME_W * 0.75 or h > FRAME_H * 0.95:
            continue

        cx = x + w / 2.0
        cy = y + h / 2.0

        # Bonus per i candidati che stanno dentro il rettangolo del foglio.
        inside_paper_rect = (px <= cx <= px + pw and py <= cy <= py + ph)
        score = area * (1.25 if inside_paper_rect else 1.0)

        candidates.append({
            "rect": (x, y, w, h),
            "area": area,
            "score": score,
            "cx": cx,
            "cy": cy,
            "dist_from_paper_center": abs(cx - paper_center_x),
        })

    if len(candidates) < 2:
        debug = cv2.cvtColor(gray_logic, cv2.COLOR_GRAY2BGR)
        debug[paper_mask > 0] = cv2.addWeighted(debug, 0.65, np.full_like(debug, (0, 80, 0)), 0.35, 0)[paper_mask > 0]
        debug[merged > 0] = (0, 0, 255)
        raise RuntimeError(
            f"Pedali non trovati: candidati={len(candidates)}. "
            "Togli i piedi dal foglio, aumenta luce uniforme, controlla che i pedali neri siano ben visibili."
        )

    # Prende i candidati più grandi, ma evita duplicati sovrapposti.
    candidates = sorted(candidates, key=lambda d: d["score"], reverse=True)

    selected = []
    for cand in candidates:
        x, y, w, h = cand["rect"]
        cx = cand["cx"]

        too_close = False
        for old in selected:
            ox, oy, ow, oh = old["rect"]
            ocx = old["cx"]
            if abs(cx - ocx) < min(w, ow) * 0.55:
                too_close = True
                break

        if not too_close:
            selected.append(cand)

        if len(selected) == 2:
            break

    if len(selected) < 2:
        raise RuntimeError(
            "Ho trovato zone nere, ma non riesco a separare due pedali. "
            "Aumenta la distanza tra i pedali nella stampa o allontana leggermente la camera."
        )

    # Non assegnare più FRENO/GAS da sinistra/destra:
    # con webcam specchiate o stampa ruotata può invertirli.
    # La classificazione robusta è per forma:
    #   GAS   = componente più alto e stretto
    #   FRENO = componente più largo e basso
    def aspect_h_over_w(cand):
        x, y, w, h = cand["rect"]
        return h / max(1.0, float(w))

    gas_cand = max(selected, key=aspect_h_over_w)
    brake_cand = min(selected, key=aspect_h_over_w)

    if FORCE_SWAP_GAS_BRAKE:
        gas_cand, brake_cand = brake_cand, gas_cand

    brake_roi = expand_roi(brake_cand["rect"])
    gas_roi = expand_roi(gas_cand["rect"])

    debug = cv2.cvtColor(gray_logic, cv2.COLOR_GRAY2BGR)
    debug[paper_mask > 0] = cv2.addWeighted(
        debug, 0.70, np.full_like(debug, (0, 60, 0)), 0.30, 0
    )[paper_mask > 0]
    debug[merged > 0] = (0, 0, 255)

    bx, by, bw, bh = brake_roi
    gx, gy, gw, gh = gas_roi
    cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)
    cv2.putText(debug, "FRENO", (bx, max(18, by - 6)), FONT, 0.65, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.rectangle(debug, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
    cv2.putText(debug, "GAS", (gx, max(18, gy - 6)), FONT, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

    return brake_roi, gas_roi, debug, paper_found


# ============================================================
# SOGLIA NERO / FACE MASK
# ============================================================
def threshold_black_calibration(gray):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    otsu_thr, _ = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    black_thr = int(np.clip(
        otsu_thr + BLACK_THRESHOLD_MARGIN,
        BLACK_THRESHOLD_MIN,
        BLACK_THRESHOLD_MAX
    ))

    black = np.zeros_like(gray, dtype=np.uint8)
    black[gray <= black_thr] = 255
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return black_thr, black


def threshold_black_live(gray, black_thr):
    black = np.zeros_like(gray, dtype=np.uint8)
    black[gray <= black_thr] = 255
    black = cv2.morphologyEx(black, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return black


def largest_true_segment(flags):
    best = None
    start = None

    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            seg = (start, i - 1)
            if best is None or seg[1] - seg[0] > best[1] - best[0]:
                best = seg
            start = None

    if start is not None:
        seg = (start, len(flags) - 1)
        if best is None or seg[1] - seg[0] > best[1] - best[0]:
            best = seg

    return best


def auto_detect_face_mask(ref_black):
    """
    Esclude automaticamente il gambo.
    La faccia del pedale è la zona dove la forma nera è più larga in orizzontale.
    """
    h, w = ref_black.shape

    work = cv2.morphologyEx(ref_black, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))
    work = cv2.dilate(work, np.ones((9, 9), np.uint8), iterations=1)

    spans = np.zeros(h, dtype=np.float32)
    for y in range(h):
        xs = np.where(work[y] > 0)[0]
        if xs.size:
            spans[y] = xs[-1] - xs[0] + 1

    if spans.max() < 10:
        return np.ones_like(ref_black) * 255, (0, 0, w, h)

    spans = np.convolve(spans, np.ones(9, dtype=np.float32) / 9, mode="same")
    broad_rows = spans > spans.max() * FACE_ROW_SPAN_RATIO
    seg = largest_true_segment(broad_rows)

    if seg is None or seg[1] - seg[0] < max(20, int(h * 0.12)):
        return np.ones_like(ref_black) * 255, (0, 0, w, h)

    y1, y2 = seg
    band = work[y1:y2 + 1]
    _, xs = np.where(band > 0)

    if xs.size == 0:
        return np.ones_like(ref_black) * 255, (0, 0, w, h)

    x1 = max(0, int(xs.min()) - FACE_PAD_X)
    x2 = min(w - 1, int(xs.max()) + FACE_PAD_X)
    y1 = max(0, int(y1) - FACE_PAD_Y)
    y2 = min(h - 1, int(y2) + FACE_PAD_Y)

    mask = np.zeros_like(ref_black)
    mask[y1:y2 + 1, x1:x2 + 1] = 255
    return mask, (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


# ============================================================
# DASHBOARD
# ============================================================
def norm_vis(img):
    if img is None:
        return np.zeros((PANEL_H, PANEL_W), dtype=np.uint8)
    arr = img.copy()
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    if arr.dtype != np.uint8:
        if arr.max() <= 1:
            arr *= 255
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def to_bgr(img):
    img = norm_vis(img)
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def panel(img, title, subtitle=""):
    img = cv2.resize(to_bgr(img), (PANEL_W, PANEL_H))
    canvas = np.zeros((PANEL_H + HEADER_H + FOOTER_H, PANEL_W, 3), dtype=np.uint8)
    canvas[HEADER_H:HEADER_H + PANEL_H] = img

    cv2.putText(canvas, title, (8, 16), FONT, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(
            canvas, subtitle,
            (8, HEADER_H + PANEL_H + 16),
            FONT, 0.42, (230, 230, 230), 1, cv2.LINE_AA
        )
    return canvas


def grid(panels, cols):
    blank = panel(np.zeros((PANEL_H, PANEL_W), dtype=np.uint8), "", "")
    while len(panels) % cols:
        panels.append(blank)
    rows = []
    for i in range(0, len(panels), cols):
        rows.append(np.hstack(panels[i:i + cols]))
    return np.vstack(rows)


# ============================================================
# STATO PEDALE
# ============================================================
def make_state(name, roi, color, hard_threshold, full_pressure_at):
    return {
        "name": name,
        "roi": roi,
        "color": color,
        "hard_threshold": hard_threshold,
        "full_pressure_at": full_pressure_at,
        "ready": False,

        "ref_gray": None,
        "black_threshold": 80,
        "face_mask": None,
        "face_rect": None,
        "ref_black": None,
        "ref_edges": None,
        "black_valid": False,
        "edge_valid": False,

        "curr_black": None,
        "curr_edges": None,
        "black_missing": None,
        "edge_missing": None,
        "diff": None,
        "changed": None,

        "black_raw": 0.0,
        "edge_raw": 0.0,
        "int_raw": 0.0,
        "combined_raw": 0.0,
        "pressure": 0.0,
        "smoothed": 0.0,
        "output": 0.0,
    }


def empty_state(name, color, hard_threshold, full_pressure_at):
    return make_state(name, None, color, hard_threshold, full_pressure_at)


def reset_states():
    brake = empty_state("FRENO", (255, 0, 0), BRAKE_HARD_THRESHOLD, BRAKE_FULL_PRESSURE_AT)
    gas = empty_state("GAS", (0, 255, 0), GAS_HARD_THRESHOLD, GAS_FULL_PRESSURE_AT)
    return brake, gas


def build_reference(samples, state):
    ref_gray = np.median(np.stack(samples), axis=0).astype(np.uint8)

    black_thr, ref_black_all = threshold_black_calibration(ref_gray)

    # Per trovare la faccia del pedale uso una versione pulita geometricamente:
    # se nella ROI entra una fuga della piastrella, non deve allargare la face mask.
    face_source = remove_long_thin_noise(ref_black_all)
    face_mask, face_rect = auto_detect_face_mask(face_source)

    ref_black = cv2.bitwise_and(ref_black_all, ref_black_all, mask=face_mask)
    ref_edges_all = get_edges(ref_gray)
    ref_edges = cv2.bitwise_and(ref_edges_all, ref_edges_all, mask=face_mask)

    black_count = cv2.countNonZero(ref_black)
    edge_count = cv2.countNonZero(ref_edges)

    state["ready"] = True
    state["ref_gray"] = ref_gray
    state["black_threshold"] = black_thr
    state["face_mask"] = face_mask
    state["face_rect"] = face_rect
    state["ref_black"] = ref_black
    state["ref_edges"] = ref_edges
    state["black_valid"] = black_count >= MIN_REF_BLACK_PIXELS
    state["edge_valid"] = edge_count >= MIN_REF_EDGE_PIXELS

    z = np.zeros_like(ref_gray)
    state["curr_black"] = z.copy()
    state["curr_edges"] = z.copy()
    state["black_missing"] = z.copy()
    state["edge_missing"] = z.copy()
    state["diff"] = z.copy()
    state["changed"] = z.copy()

    for k in ["black_raw", "edge_raw", "int_raw", "combined_raw", "pressure", "smoothed", "output"]:
        state[k] = 0.0

    print(
        f"[{state['name']}] ROI={state['roi']} thr={black_thr} "
        f"black={black_count} valid={state['black_valid']} "
        f"edges={edge_count} valid={state['edge_valid']} "
        f"face_rect={face_rect}"
    )

    return state


# ============================================================
# MISURE PRESSIONE
# ============================================================
def measure_black_missing(state, curr_black):
    ref_black = state["ref_black"]
    total = cv2.countNonZero(ref_black)

    if total < MIN_REF_BLACK_PIXELS:
        return 0.0, np.zeros_like(state["face_mask"])

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (EDGE_TOLERANCE_RADIUS * 2 + 1, EDGE_TOLERANCE_RADIUS * 2 + 1)
    )

    curr_dilated = cv2.dilate(curr_black, k, iterations=1)
    visible = cv2.bitwise_and(ref_black, curr_dilated)
    missing = cv2.bitwise_and(ref_black, cv2.bitwise_not(visible))

    raw = cv2.countNonZero(missing) / total
    return float(np.clip(raw, 0.0, 1.0)), missing


def measure_edge_missing(state, curr_edges):
    ref_edges = state["ref_edges"]
    total = cv2.countNonZero(ref_edges)

    if total < MIN_REF_EDGE_PIXELS:
        return 0.0, np.zeros_like(state["face_mask"])

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (EDGE_TOLERANCE_RADIUS * 2 + 1, EDGE_TOLERANCE_RADIUS * 2 + 1)
    )

    curr_dilated = cv2.dilate(curr_edges, k, iterations=1)
    visible = cv2.bitwise_and(ref_edges, curr_dilated)
    missing = cv2.bitwise_and(ref_edges, cv2.bitwise_not(visible))

    raw = cv2.countNonZero(missing) / total
    return float(np.clip(raw, 0.0, 1.0)), missing


def measure_intensity_changed(state, curr_gray):
    ref_gray = state["ref_gray"]
    mask = state["face_mask"] > 0

    if np.count_nonzero(mask) == 0:
        z = np.zeros_like(state["face_mask"])
        return 0.0, z, z

    ref = ref_gray.astype(np.float32)
    curr = curr_gray.astype(np.float32)

    # Compensa variazioni globali di luminosità nella faccia del pedale.
    ref_norm = ref - np.median(ref[mask])
    curr_norm = curr - np.median(curr[mask])

    diff = np.abs(ref_norm - curr_norm)
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)

    changed = np.zeros_like(state["face_mask"], dtype=np.uint8)
    changed[(diff > INTENSITY_DIFF_THRESHOLD) & mask] = 255

    changed = cv2.medianBlur(changed, 5)
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    raw = cv2.countNonZero(changed) / np.count_nonzero(mask)
    diff_vis = cv2.bitwise_and(diff_u8, diff_u8, mask=state["face_mask"])

    return float(np.clip(raw, 0.0, 1.0)), diff_vis, changed


def combine_signals(state, black_raw, edge_raw, int_raw):
    parts = []

    if state["black_valid"]:
        parts.append((BLACK_MISSING_WEIGHT, black_raw))

    if state["edge_valid"]:
        parts.append((EDGE_MISSING_WEIGHT, edge_raw))

    # Il segnale intensity resta sempre come fallback.
    parts.append((INTENSITY_WEIGHT, int_raw))

    total_w = sum(w for w, _ in parts)
    return float(np.clip(sum(w * v for w, v in parts) / total_w, 0.0, 1.0))


def update_state(state, live_gray):
    if not state["ready"] or state["roi"] is None:
        return

    curr_black_all = threshold_black_live(live_gray, state["black_threshold"])
    curr_black = cv2.bitwise_and(curr_black_all, curr_black_all, mask=state["face_mask"])

    curr_edges_all = get_edges(live_gray)
    curr_edges = cv2.bitwise_and(curr_edges_all, curr_edges_all, mask=state["face_mask"])

    black_raw, black_missing = measure_black_missing(state, curr_black)
    edge_raw, edge_missing = measure_edge_missing(state, curr_edges)
    int_raw, diff, changed = measure_intensity_changed(state, live_gray)

    combined_raw = combine_signals(state, black_raw, edge_raw, int_raw)
    pressure = map_pressure(combined_raw, state["hard_threshold"], state["full_pressure_at"])

    state["black_raw"] = black_raw
    state["edge_raw"] = edge_raw
    state["int_raw"] = int_raw
    state["combined_raw"] = combined_raw
    state["pressure"] = pressure
    state["smoothed"] = smooth(state["smoothed"], pressure)
    state["output"] = response_curve(state["smoothed"])

    state["curr_black"] = curr_black
    state["curr_edges"] = curr_edges
    state["black_missing"] = black_missing
    state["edge_missing"] = edge_missing
    state["diff"] = diff
    state["changed"] = changed


# ============================================================
# VISUALIZZAZIONE
# ============================================================
def make_overlay(live_gray, state):
    overlay = cv2.cvtColor(live_gray, cv2.COLOR_GRAY2BGR)

    if not state["ready"]:
        return overlay

    contours, _ = cv2.findContours(state["face_mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)

    # blu     = nero di riferimento
    # giallo  = nero attuale
    # rosso   = nero sparito
    # magenta = edge sparito
    overlay[state["ref_black"] > 0] = (255, 0, 0)
    overlay[state["curr_black"] > 0] = (0, 255, 255)
    overlay[state["black_missing"] > 0] = (0, 0, 255)
    overlay[state["edge_missing"] > 0] = (255, 0, 255)

    return overlay


def draw_bar(frame, x, y, value, label, color):
    w, h = 180, 18
    cv2.rectangle(frame, (x, y), (x + w, y + h), (40, 40, 40), -1)
    cv2.rectangle(frame, (x, y), (x + int(w * value), y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (230, 230, 230), 1)
    cv2.putText(frame, f"{label}: {int(value * 100)}%", (x, y - 5), FONT, 0.5, color, 1, cv2.LINE_AA)


def draw_output(frame, brake_state, gas_state, calibrated):
    for st in [brake_state, gas_state]:
        if st["roi"] is None:
            continue

        x, y, w, h = clamp_roi(st["roi"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), st["color"], 2)
        cv2.putText(frame, st["name"], (x, max(18, y - 6)), FONT, 0.55, st["color"], 2, cv2.LINE_AA)

        if st["ready"] and st["face_rect"] is not None:
            fx, fy, fw, fh = st["face_rect"]
            cv2.rectangle(frame, (x + fx, y + fy), (x + fx + fw, y + fy + fh), (0, 255, 255), 2)

    draw_bar(frame, 20, 30, gas_state["output"], "GAS", (0, 255, 0))
    draw_bar(frame, 20, 65, brake_state["output"], "FRENO", (255, 0, 0))

    if calibrated:
        cv2.putText(
            frame,
            f"G black={gas_state['black_raw']:.2f} edge={gas_state['edge_raw']:.2f} int={gas_state['int_raw']:.2f} raw={gas_state['combined_raw']:.2f}",
            (20, 110), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"F black={brake_state['black_raw']:.2f} edge={brake_state['edge_raw']:.2f} int={brake_state['int_raw']:.2f} raw={brake_state['combined_raw']:.2f}",
            (20, 130), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame,
            "Premi C senza piedi sul foglio: auto ROI + calibrazione",
            (20, 115), FONT, 0.55, (0, 255, 255), 2, cv2.LINE_AA
        )

    cv2.putText(
        frame,
        "C auto-calibra | R reset | S screenshot | Q esci",
        (20, FRAME_H - 14), FONT, 0.50, (0, 255, 255), 1, cv2.LINE_AA
    )


def global_grid(output, gray_raw, gray_norm, gray_clahe, gray_logic, auto_debug=None):
    auto_panel = auto_debug if auto_debug is not None else get_edges(gray_logic)
    auto_title = "AUTO ROI DEBUG" if auto_debug is not None else "EDGES GLOBALI"

    return grid([
        panel(output, "OUTPUT", "giallo = face mask"),
        panel(gray_raw, "GRAY RAW"),
        panel(gray_norm, "GRAY NORM", "bianco verso 255"),
        panel(gray_clahe, "CLAHE DEBUG", "solo visuale"),
        panel(auto_panel, auto_title),
    ], 5)


def pedal_grid(st, live_gray=None):
    name = st["name"]

    if not st["ready"] or live_gray is None:
        z = np.zeros((PANEL_H, PANEL_W), dtype=np.uint8)
        return grid([
            panel(z, f"{name} ROI", "non calibrato"),
            panel(z, f"{name} FACE", "premi C"),
            panel(z, f"{name} REF BLACK"),
            panel(z, f"{name} CURR BLACK"),
            panel(z, f"{name} BLACK MISS"),
            panel(z, f"{name} REF EDGES"),
            panel(z, f"{name} CURR EDGES"),
            panel(z, f"{name} EDGE MISS"),
            panel(z, f"{name} DIFF"),
            panel(z, f"{name} OVERLAY"),
        ], 5)

    ref_black_count = cv2.countNonZero(st["ref_black"])
    ref_edge_count = cv2.countNonZero(st["ref_edges"])
    black_status = "on" if st["black_valid"] else "off"
    edge_status = "on" if st["edge_valid"] else "off"

    return grid([
        panel(live_gray, f"{name} ROI", f"thr={st['black_threshold']}"),
        panel(st["face_mask"], f"{name} FACE", f"rect={st['face_rect']}"),
        panel(st["ref_black"], f"{name} REF BLACK", f"{ref_black_count}px {black_status}"),
        panel(st["curr_black"], f"{name} CURR BLACK", f"b={st['black_raw']:.2f}"),
        panel(st["black_missing"], f"{name} BLACK MISS"),
        panel(st["ref_edges"], f"{name} REF EDGES", f"{ref_edge_count}px {edge_status}"),
        panel(st["curr_edges"], f"{name} CURR EDGES", f"e={st['edge_raw']:.2f}"),
        panel(st["edge_missing"], f"{name} EDGE MISS"),
        panel(st["diff"], f"{name} DIFF", f"i={st['int_raw']:.2f}"),
        panel(make_overlay(live_gray, st), f"{name} OVERLAY", f"raw={st['combined_raw']:.2f} out={st['output']:.2f}"),
    ], 5)


# ============================================================
# CALIBRAZIONE COMPLETA: AUTO ROI + REFERENCE
# ============================================================
def try_lock_camera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    try:
        # Queste property non funzionano su tutte le webcam, ma se funzionano aiutano.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    except Exception:
        pass


def calibrate_auto(cap):
    print("Auto-calibrazione: togli i piedi dal foglio.")
    print("Raccolgo alcuni frame e rilevo automaticamente FRENO/GAS...")

    frames_logic = []
    frames_color = []

    for i in range(CALIBRATION_FRAMES):
        ret, frame = cap.read()
        if not ret:
            continue

        frame, _, _, _, gray_logic = preprocess(frame)
        frames_logic.append(gray_logic)
        frames_color.append(frame)

        preview = frame.copy()
        cv2.putText(
            preview,
            f"Auto-calibrazione... {i + 1}/{CALIBRATION_FRAMES} - niente piedi sul foglio",
            (20, 35), FONT, 0.65, (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imshow(WINDOW_NAME, preview)
        cv2.waitKey(25)

    if len(frames_logic) < 5:
        raise RuntimeError("Troppi pochi frame per calibrare.")

    median_logic = np.median(np.stack(frames_logic), axis=0).astype(np.uint8)
    brake_roi, gas_roi, auto_debug, paper_found = auto_detect_pedal_rois(median_logic)

    print("ROI rilevate automaticamente:")
    print("  FRENO:", brake_roi)
    print("  GAS:  ", gas_roi)
    print("  Foglio trovato:", paper_found)

    brake_state = make_state("FRENO", brake_roi, (255, 0, 0), BRAKE_HARD_THRESHOLD, BRAKE_FULL_PRESSURE_AT)
    gas_state = make_state("GAS", gas_roi, (0, 255, 0), GAS_HARD_THRESHOLD, GAS_FULL_PRESSURE_AT)

    brake_samples = [crop(g, brake_roi) for g in frames_logic]
    gas_samples = [crop(g, gas_roi) for g in frames_logic]

    brake_state = build_reference(brake_samples, brake_state)
    gas_state = build_reference(gas_samples, gas_state)

    # Mostra per un attimo le ROI trovate.
    result = frames_color[-1].copy()
    draw_output(result, brake_state, gas_state, calibrated=True)
   # cv2.imshow("Pedal Vision Debug", result)
    cv2.waitKey(500)

    print("Auto-calibrazione completata.")
    return brake_state, gas_state, auto_debug


# ============================================================
# MAIN
# ============================================================
cap = cv2.VideoCapture(CAMERA_ID,cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Webcam non trovata.")

try_lock_camera(cap)

brake_state, gas_state = reset_states()
auto_debug_last = None
last_dashboard = None

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("Comandi: C auto-calibra | R reset | S screenshot | Q esci")
print("All'avvio premi C con il foglio visibile e i piedi lontani.")
print("SHOW_DEBUG_DASHBOARD =", SHOW_DEBUG_DASHBOARD)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame non letto.")
        break

    frame, gray_raw, gray_norm, gray_clahe, gray_logic = preprocess(frame)

    calibrated = brake_state["ready"] and gas_state["ready"]

    brake_live = None
    gas_live = None

    if calibrated:
        brake_live = crop(gray_logic, brake_state["roi"])
        gas_live = crop(gray_logic, gas_state["roi"])

        update_state(brake_state, brake_live)
        update_state(gas_state, gas_live)

    # Schermata principale a colori
    output = frame.copy()
    draw_output(output, brake_state, gas_state, calibrated=calibrated)

    # ========================================================
    # DISPLAY
    # ========================================================
    if SHOW_DEBUG_DASHBOARD:
        dashboard = np.vstack([
            global_grid(
                output,
                gray_raw,
                gray_norm,
                gray_clahe,
                gray_logic,
                auto_debug=auto_debug_last
            ),
            pedal_grid(brake_state, brake_live),
            pedal_grid(gas_state, gas_live),
        ])

        display = dashboard
        last_dashboard = dashboard

    else:
        if OUTPUT_SCALE != 1.0:
            display = cv2.resize(
                output,
                None,
                fx=OUTPUT_SCALE,
                fy=OUTPUT_SCALE,
                interpolation=cv2.INTER_LINEAR
            )
        else:
            display = output

        last_dashboard = display

    cv2.imshow(WINDOW_NAME, display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("c"):
        try:
            brake_state, gas_state, auto_debug_last = calibrate_auto(cap)
        except Exception as e:
            print("ERRORE calibrazione:", e)
            brake_state, gas_state = reset_states()
            auto_debug_last = None

    elif key == ord("r"):
        brake_state, gas_state = reset_states()
        auto_debug_last = None
        print("Reset completato.")

    elif key == ord("s"):
        if last_dashboard is not None:
            if SHOW_DEBUG_DASHBOARD:
                filename = "pedal_debug_dashboard.png"
            else:
                filename = "pedal_output.png"

            cv2.imwrite(filename, last_dashboard)
            print("Screenshot salvato:", filename)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()