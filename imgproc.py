"""imgproc.py — Modul pemrosesan citra untuk EdgeVision."""
import cv2
import numpy as np
import time

# ── Haar Cascade classifiers ─────────────────────────────────────────────────
_CASCADE_DIR  = cv2.data.haarcascades
FACE_CASCADE  = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_frontalface_default.xml")
EYE_CASCADE   = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_eye.xml")
SMILE_CASCADE = cv2.CascadeClassifier(_CASCADE_DIR + "haarcascade_smile.xml")

MAX_DIM = 1024  # Batas panjang sisi terpanjang setelah auto-resize


# ─────────────────────────────────────────────────────────────────────────────
#  Utilitas dasar
# ─────────────────────────────────────────────────────────────────────────────

def load_image(file_bytes):
    """
    Muat gambar dari bytes (hasil upload/kamera).
    Auto-resize jika dimensi melebihi MAX_DIM.
    Mengembalikan array BGR atau None jika file rusak.
    """
    try:
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale   = MAX_DIM / max(h, w)
            new_w   = int(w * scale)
            new_h   = int(h * scale)
            img     = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img
    except Exception:
        return None


def to_rgb(img_bgr):
    """Konversi BGR → RGB (untuk ditampilkan)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_gray(img_bgr):
    """Konversi BGR → Grayscale."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ─────────────────────────────────────────────────────────────────────────────
#  Deteksi tepi
# ─────────────────────────────────────────────────────────────────────────────

def apply_sobel(gray, ksize=3):
    """
    Deteksi tepi dengan Operator Sobel.
    Mengembalikan (gambar_tepi_RGB uint8, waktu_komputasi_ms float).
    """
    t0  = time.perf_counter()
    gx  = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy  = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if mag.max() > 0:
        mag = np.uint8(255.0 * mag / mag.max())
    else:
        mag = np.zeros_like(gray, dtype=np.uint8)
    t1  = time.perf_counter()
    return cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB), (t1 - t0) * 1000.0


def apply_canny(gray, low=50, high=150):
    """
    Deteksi tepi dengan Algoritma Canny.
    Mengembalikan (gambar_tepi_RGB uint8, waktu_komputasi_ms float).
    """
    t0    = time.perf_counter()
    edges = cv2.Canny(gray, low, high)
    t1    = time.perf_counter()
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), (t1 - t0) * 1000.0


def edge_density(edge_rgb):
    """
    Persentase piksel tepi terdeteksi terhadap total piksel (0.0–100.0).
    Input: gambar RGB hasil apply_sobel atau apply_canny.
    """
    gray = cv2.cvtColor(edge_rgb, cv2.COLOR_RGB2GRAY)
    return 100.0 * float(np.count_nonzero(gray)) / float(gray.size)


# ─────────────────────────────────────────────────────────────────────────────
#  Deteksi fitur wajah
# ─────────────────────────────────────────────────────────────────────────────

def detect_face(img_bgr):
    """
    Deteksi otomatis: wajah → mata (area atas 55%) → mulut (area bawah 40%).

    Mengembalikan dict dengan kunci:
      annotated  : ndarray RGB  — gambar anotasi bounding box
      face       : ndarray RGB atau None — crop wajah terbesar
      right_eye  : ndarray RGB atau None — mata kanan subjek (kiri di gambar)
      left_eye   : ndarray RGB atau None — mata kiri subjek (kanan di gambar)
      mouth      : ndarray RGB atau None — area mulut
      notes      : list[str]  — catatan keterbatasan deteksi
    """
    gray   = to_gray(img_bgr)
    result = {
        "annotated": to_rgb(img_bgr),
        "face":      None,
        "right_eye": None,
        "left_eye":  None,
        "mouth":     None,
        "notes":     [],
    }

    # ── 1. Deteksi wajah ────────────────────────────────────────────────────
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )
    if len(faces) == 0:
        result["notes"].append(
            "Wajah tidak terdeteksi — gunakan foto dengan cahaya merata "
            "dan wajah menghadap lurus ke kamera."
        )
        return result

    # Ambil wajah dengan area terbesar
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_bgr    = img_bgr[y : y + h, x : x + w]
    face_gray   = gray[y : y + h, x : x + w]
    result["face"] = to_rgb(face_bgr)

    # ── 2. Deteksi mata — dibatasi di 55% area atas wajah ──────────────────
    eye_h      = int(h * 0.55)
    eye_region = face_gray[:eye_h, :]
    eyes       = []          # untuk anotasi

    raw_eyes = EYE_CASCADE.detectMultiScale(
        eye_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    if len(raw_eyes) > 0:
        # Urutkan kiri→kanan di gambar, ambil maks 2
        eyes = sorted(raw_eyes.tolist(), key=lambda e: e[0])[:2]
        crops = []
        for ex, ey, ew, eh in eyes:
            pad  = int(ew * 0.3)
            x1   = max(0, ex - pad)
            y1   = max(0, ey - pad)
            x2   = min(face_bgr.shape[1], ex + ew + pad)
            y2   = min(eye_h, ey + eh + pad)
            c    = face_bgr[y1:y2, x1:x2]
            if c.size > 0:
                crops.append(to_rgb(c))

        # Mata paling kiri di gambar = mata KANAN subjek
        if len(crops) >= 2:
            result["right_eye"] = crops[0]
            result["left_eye"]  = crops[1]
        elif len(crops) == 1:
            result["right_eye"] = crops[0]
    else:
        result["notes"].append(
            "Mata tidak terdeteksi otomatis — pencahayaan atau sudut wajah "
            "tidak optimal untuk Haar Eye Cascade."
        )

    # ── 3. Deteksi mulut — dibatasi di 40% area bawah wajah ───────────────
    mouth_y0   = int(h * 0.60)
    mouth_face = face_bgr[mouth_y0:, :]
    mouth_gray = face_gray[mouth_y0:, :]
    smiles     = []

    raw_smiles = SMILE_CASCADE.detectMultiScale(
        mouth_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 15)
    )
    if len(raw_smiles) > 0:
        smiles      = sorted(raw_smiles.tolist(), key=lambda s: s[2] * s[3], reverse=True)
        sx, sy, sw, sh = smiles[0]
        pad  = int(sw * 0.2)
        x1   = max(0, sx - pad)
        y1   = max(0, sy - pad)
        x2   = min(mouth_face.shape[1], sx + sw + pad)
        y2   = min(mouth_face.shape[0], sy + sh + pad)
        c    = mouth_face[y1:y2, x1:x2]
        if c.size > 0:
            result["mouth"] = to_rgb(c)
    else:
        # Fallback: estimasi proporsional berdasarkan geometri wajah
        my1  = max(0, int(h * 0.68) - mouth_y0)
        my2  = min(mouth_face.shape[0], int(h * 0.88) - mouth_y0)
        mx1  = max(0, int(w * 0.20))
        mx2  = min(mouth_face.shape[1], int(w * 0.80))
        c    = mouth_face[my1:my2, mx1:mx2]
        if c.size > 0:
            result["mouth"] = to_rgb(c)
        result["notes"].append(
            "Area mulut diestimasi secara proporsional karena senyuman "
            "tidak terdeteksi (Haar Smile Cascade memerlukan senyum yang jelas)."
        )

    # ── 4. Anotasi bounding box ──────────────────────────────────────────────
    ann = img_bgr.copy()
    cv2.rectangle(ann, (x, y), (x + w, y + h), (0, 210, 255), 2)
    cv2.putText(ann, "WAJAH", (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 210, 255), 1, cv2.LINE_AA)

    for ex, ey, ew, eh in eyes:
        cv2.rectangle(ann, (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                      (255, 80, 0), 2)

    if len(smiles) > 0:
        sx, sy, sw, sh = smiles[0]
        cv2.rectangle(ann,
                      (x + sx, y + mouth_y0 + sy),
                      (x + sx + sw, y + mouth_y0 + sy + sh),
                      (0, 255, 80), 2)
        cv2.putText(ann, "MULUT",
                    (x + sx, max(0, y + mouth_y0 + sy - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 80), 1, cv2.LINE_AA)

    result["annotated"] = to_rgb(ann)
    return result
