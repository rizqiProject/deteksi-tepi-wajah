"""
imgproc.py
Modul inti untuk penelitian "Perbandingan Kinerja Operator Sobel dan
Algoritma Canny dalam Deteksi Tepi pada Citra Wajah".

Berisi:
- Konversi grayscale
- Operator Sobel (magnitudo gradien)
- Algoritma Canny
- Deteksi otomatis wajah, mata, dan area mulut (Haar Cascade)

Catatan metodologis (biar konsisten dengan Bab 3 makalah):
- Semua operator tepi diterapkan pada citra GRAYSCALE, bukan RGB langsung,
  sesuai penjelasan di Landasan Teori.
- Crop fitur spesifik (mata/mulut) diambil dari foto dasar yang sama
  dengan kontur wajah keseluruhan, untuk menjaga konsistensi pencahayaan.
"""

import os

import cv2
import numpy as np

# File cascade (.xml) disertakan langsung sejajar dengan file ini di dalam
# project (bukan di subfolder), supaya tidak bergantung pada
# cv2.data.haarcascades yang kadang tidak tersedia/berbeda-beda di
# lingkungan server (mis. Streamlit Cloud), dan supaya upload di GitHub
# tetap simpel (tidak perlu bikin folder).
_CASCADE_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Dasar: grayscale, Sobel, Canny
# ---------------------------------------------------------------------------

def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Konversi citra RGB/BGR ke grayscale (satu kanal intensitas)."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def apply_sobel(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Operator Sobel: menghitung Gx dan Gy lalu magnitudo gradien
    G = sqrt(Gx^2 + Gy^2), dinormalisasi ke rentang 0-255 agar bisa
    ditampilkan sebagai citra 8-bit.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = np.uint8(255 * magnitude / (magnitude.max() + 1e-8))
    return magnitude


def apply_canny(gray: np.ndarray, low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
    """
    Algoritma Canny: Gaussian blur -> gradien -> non-max suppression ->
    hysteresis thresholding (dua tahap terakhir sudah built-in di cv2.Canny).
    Blur manual ditambahkan dulu supaya tahap 'reduksi derau' eksplisit
    terlihat di kode, sesuai penjelasan 4 tahap di Bab 2.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.4)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


# ---------------------------------------------------------------------------
# Deteksi otomatis wajah, mata, mulut (untuk crop fitur spesifik)
# ---------------------------------------------------------------------------

_face_cascade = cv2.CascadeClassifier(os.path.join(_CASCADE_DIR, "haarcascade_frontalface_default.xml"))
_eye_cascade = cv2.CascadeClassifier(os.path.join(_CASCADE_DIR, "haarcascade_eye.xml"))
_smile_cascade = cv2.CascadeClassifier(os.path.join(_CASCADE_DIR, "haarcascade_smile.xml"))

for _name, _clf in [("face", _face_cascade), ("eye", _eye_cascade), ("smile", _smile_cascade)]:
    if _clf.empty():
        raise RuntimeError(
            f"Gagal memuat cascade '{_name}'. Pastikan folder 'cascades/' ikut ter-upload ke repo."
        )


def detect_face(gray: np.ndarray):
    """Kembalikan bounding box wajah terbesar (x, y, w, h), atau None."""
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # ambil wajah dengan area terbesar (paling dominan di foto)
    return max(faces, key=lambda f: f[2] * f[3])


def detect_eyes(gray_face_roi: np.ndarray):
    """Deteksi mata di dalam ROI wajah. Kembalikan list (x, y, w, h) relatif ROI."""
    eyes = _eye_cascade.detectMultiScale(gray_face_roi, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20))
    return list(eyes)


def detect_mouth(gray_face_roi: np.ndarray, face_h: int):
    """
    Deteksi area mulut memakai haarcascade_smile pada separuh bawah wajah
    (mengurangi false positive di area mata/hidung). Jika gagal terdeteksi,
    fallback ke estimasi proporsional (heuristik) berdasarkan anatomi wajah
    umum: mulut ada di sekitar 65%-95% tinggi wajah dari atas.
    """
    lower_half_y = face_h // 2
    lower_roi = gray_face_roi[lower_half_y:, :]
    mouths = _smile_cascade.detectMultiScale(lower_roi, scaleFactor=1.7, minNeighbors=20, minSize=(25, 15))
    if len(mouths) > 0:
        mx, my, mw, mh = max(mouths, key=lambda m: m[2] * m[3])
        return (mx, my + lower_half_y, mw, mh), "haar_smile"
    # fallback heuristik proporsional (dicatat sebagai keterbatasan di makalah)
    fw = gray_face_roi.shape[1]
    fh = gray_face_roi.shape[0]
    est_x = int(0.25 * fw)
    est_y = int(0.65 * fh)
    est_w = int(0.5 * fw)
    est_h = int(0.30 * fh)
    return (est_x, est_y, est_w, est_h), "fallback_proporsional"


def extract_face_and_features(image_bgr: np.ndarray):
    """
    Pipeline lengkap: deteksi wajah -> crop kontur wajah keseluruhan ->
    deteksi mata & mulut di dalam wajah -> crop tiap fitur.

    Return dict berisi crop grayscale untuk: wajah_utuh, mata (list), mulut,
    beserta metadata metode deteksi (untuk ditulis di Bab 3/4 sebagai
    transparansi metodologis).
    """
    gray = to_grayscale(image_bgr)
    face_box = detect_face(gray)

    result = {
        "gray_full": gray,
        "face_box": face_box,
        "face_crop": None,
        "eyes_crop": [],
        "mouth_crop": None,
        "mouth_method": None,
    }

    if face_box is None:
        return result  # tidak ada wajah terdeteksi -> caller harus menangani

    fx, fy, fw, fh = face_box
    face_roi = gray[fy:fy + fh, fx:fx + fw]
    result["face_crop"] = face_roi

    eyes = detect_eyes(face_roi)
    for (ex, ey, ew, eh) in eyes:
        result["eyes_crop"].append(face_roi[ey:ey + eh, ex:ex + ew])

    (mx, my, mw, mh), method = detect_mouth(face_roi, fh)
    result["mouth_crop"] = face_roi[my:my + mh, mx:mx + mw]
    result["mouth_method"] = method

    return result
