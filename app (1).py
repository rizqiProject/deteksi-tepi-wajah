"""
app.py — Web app Streamlit untuk penelitian:
"Perbandingan Kinerja Operator Sobel dan Algoritma Canny dalam
Deteksi Tepi pada Citra Wajah: Studi Kasus Kontur Wajah dan Fitur
Spesifik Wajah"

Cara jalan lokal (opsional, buat coba-coba sebelum deploy):
    pip install -r requirements.txt
    streamlit run app.py

Untuk demo ke dosen, deploy ke Streamlit Community Cloud (lihat
README_DEPLOY.md) supaya dapat link publik.
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from imgproc import to_grayscale, apply_sobel, apply_canny, extract_face_and_features

st.set_page_config(page_title="Deteksi Tepi Wajah: Sobel vs Canny", layout="wide")

st.title("Perbandingan Operator Sobel & Algoritma Canny pada Citra Wajah")
st.caption(
    "Demo program penelitian mata kuliah Pengolahan Citra — "
    "membandingkan deteksi tepi pada kontur wajah keseluruhan vs. fitur wajah spesifik (mata, mulut)."
)

with st.sidebar:
    st.header("Pengaturan")
    st.subheader("Parameter Canny")
    low_th = st.slider("Threshold bawah (low)", 0, 255, 50, step=5)
    high_th = st.slider("Threshold atas (high)", 0, 255, 150, step=5)
    st.subheader("Parameter Sobel")
    ksize = st.select_slider("Ukuran kernel Sobel (ksize)", options=[1, 3, 5, 7], value=3)
    st.divider()
    st.caption(
        "Tips: kalau hasil Canny terlalu banyak noise, naikkan threshold. "
        "Kalau tepi penting hilang, turunkan threshold."
    )

source_mode = st.radio(
    "Sumber gambar",
    ["Upload dari file/galeri", "Ambil foto langsung (kamera)"],
    horizontal=True,
)

uploaded_files = []

if source_mode == "Upload dari file/galeri":
    uploaded_files = st.file_uploader(
        "Upload foto wajah (bisa lebih dari satu — variasi cahaya/ekspresi)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Silakan upload minimal 1 foto wajah (format JPG/PNG) untuk mulai.")
        st.stop()
else:
    st.caption(
        "Browser akan minta izin akses kamera. Klik tombol kamera untuk ambil foto, "
        "bisa diulang (retake) kalau hasilnya kurang pas."
    )
    camera_photo = st.camera_input("Ambil foto wajah")
    if camera_photo is None:
        st.info("Silakan ambil foto lewat kamera untuk mulai.")
        st.stop()
    uploaded_files = [camera_photo]


def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def show_row(caption_left, img_left, caption_mid, img_mid, caption_right, img_right):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(img_left, caption=caption_left, use_container_width=True, clamp=True)
    with c2:
        st.image(img_mid, caption=caption_mid, use_container_width=True, clamp=True)
    with c3:
        st.image(img_right, caption=caption_right, use_container_width=True, clamp=True)


for idx, uploaded in enumerate(uploaded_files):
    st.divider()
    st.subheader(f"Foto {idx + 1}: {uploaded.name}")

    pil_img = Image.open(uploaded)
    bgr = pil_to_bgr(pil_img)

    tab_full, tab_feature = st.tabs(["Kontur Wajah Keseluruhan", "Fitur Spesifik (Mata/Mulut)"])

    # ---------------- Kontur wajah keseluruhan ----------------
    with tab_full:
        gray_full = to_grayscale(bgr)
        sobel_full = apply_sobel(gray_full, ksize=ksize)
        canny_full = apply_canny(gray_full, low_th, high_th)
        show_row("Grayscale (asli)", gray_full, "Sobel", sobel_full, "Canny", canny_full)

    # ---------------- Fitur spesifik: mata & mulut ----------------
    with tab_feature:
        result = extract_face_and_features(bgr)

        if result["face_box"] is None:
            st.warning(
                "Wajah tidak terdeteksi otomatis pada foto ini. "
                "Coba foto dengan pencahayaan lebih merata / wajah menghadap kamera."
            )
        else:
            face_crop = result["face_crop"]
            st.markdown("**Area wajah terdeteksi (dipakai sebagai acuan crop fitur):**")
            st.image(face_crop, caption="Crop wajah (grayscale)", width=250, clamp=True)

            # Mulut
            if result["mouth_crop"] is not None and result["mouth_crop"].size > 0:
                st.markdown(
                    f"**Fitur: Mulut** _(metode deteksi: `{result['mouth_method']}`)_"
                )
                mc = result["mouth_crop"]
                sobel_m = apply_sobel(mc, ksize=ksize)
                canny_m = apply_canny(mc, low_th, high_th)
                show_row("Grayscale (crop mulut)", mc, "Sobel", sobel_m, "Canny", canny_m)
                if result["mouth_method"] == "fallback_proporsional":
                    st.caption(
                        "⚠️ Mulut tidak terdeteksi Haar Cascade langsung (umum terjadi pada ekspresi netral) "
                        "— area diestimasi memakai proporsi anatomi wajah standar. "
                        "Catat ini sebagai keterbatasan metode di Bab 4/5."
                    )

            # Mata (bisa 0, 1, atau 2 terdeteksi)
            if result["eyes_crop"]:
                st.markdown(f"**Fitur: Mata** _(terdeteksi {len(result['eyes_crop'])} area)_")
                for i, ec in enumerate(result["eyes_crop"]):
                    if ec.size == 0:
                        continue
                    sobel_e = apply_sobel(ec, ksize=ksize)
                    canny_e = apply_canny(ec, low_th, high_th)
                    show_row(
                        f"Grayscale (mata {i + 1})", ec,
                        "Sobel", sobel_e,
                        "Canny", canny_e,
                    )
            else:
                st.warning("Mata tidak terdeteksi otomatis pada foto ini.")

st.divider()
st.caption(
    "Catatan metodologis: semua operator diterapkan pada citra grayscale (bukan RGB langsung). "
    "Deteksi wajah/mata memakai Haar Cascade OpenCV; area mulut memakai haarcascade_smile dengan "
    "fallback estimasi proporsional bila tidak terdeteksi."
)
