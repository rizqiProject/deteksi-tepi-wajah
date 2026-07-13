"""app.py — EdgeVision: Deteksi Tepi Wajah | Pengolahan Citra Digital"""

import io
import base64

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from imgproc import (
    load_image,
    to_rgb,
    to_gray,
    apply_sobel,
    apply_canny,
    edge_density,
    detect_face,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Page config — wajib dipanggil pertama
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EdgeVision — Deteksi Tepi Wajah",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers rendering
# ─────────────────────────────────────────────────────────────────────────────

def _encode(arr_rgb: np.ndarray, quality: int = 88) -> str:
    """Encode array RGB uint8 → base64 WEBP untuk embed di HTML."""
    pil = Image.fromarray(arr_rgb.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def img_card(arr_rgb: np.ndarray, title: str, badge: str = "") -> str:
    """Render satu gambar sebagai HTML card dengan label bar."""
    b64      = _encode(arr_rgb)
    badge_h  = f'<span class="badge">{badge}</span>' if badge else ""
    return (
        f'<div class="ic">'
        f'  <div class="ic-h">{title}{badge_h}</div>'
        f'  <img src="data:image/webp;base64,{b64}" class="ic-img" />'
        f'</div>'
    )


def metric(label: str, val: str, unit: str = "", hi: bool = False) -> str:
    """Satu metric card HTML."""
    color = "var(--ac)" if hi else "var(--tx)"
    return (
        f'<div class="mc">'
        f'  <div class="mc-l">{label}</div>'
        f'  <div class="mc-v" style="color:{color}">{val}'
        f'    <span class="mc-u">{unit}</span>'
        f'  </div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#0D1117; --s1:#161B22; --s2:#21262D;
  --bd:#30363D; --ac:#2F81F7; --ac-dim:rgba(47,129,247,.12);
  --grn:#3FB950; --org:#E3885C;
  --tx:#E6EDF3; --mu:#8B949E;
  --font:'Space Grotesk',system-ui,sans-serif;
  --mono:'JetBrains Mono','Courier New',monospace;
}

/* ── Reset / base ── */
.stApp{background:var(--bg)!important;font-family:var(--font);color:var(--tx);}
#MainMenu,footer,[data-testid="stToolbar"]{display:none!important;}
.stApp > header{background:transparent!important;}
p,li{color:var(--tx);}

/* ── Sidebar ── */
[data-testid="stSidebar"]{background:var(--s1)!important;border-right:1px solid var(--bd);}
[data-testid="stSidebar"] p{color:var(--mu)!important;font-size:.78rem;font-family:var(--mono);}
[data-testid="stSidebar"] h3{color:var(--tx)!important;font-size:.9rem;}
[data-testid="stSidebar"] label{color:var(--mu)!important;font-size:.8rem!important;}
[data-testid="stSidebar"] hr{border-color:var(--bd)!important;}

/* ── Header ── */
.ev-hdr{padding:1.75rem 0 1.4rem;border-bottom:1px solid var(--bd);margin-bottom:1.75rem;}
.ev-eye{font-family:var(--mono);font-size:.62rem;letter-spacing:.18em;color:var(--ac);text-transform:uppercase;margin-bottom:.45rem;}
.ev-title{font-size:2.1rem;font-weight:700;letter-spacing:-.03em;color:var(--tx);margin:0;line-height:1.1;}
.ev-title em{color:var(--ac);font-style:normal;}
.ev-sub{color:var(--mu);font-size:.88rem;margin-top:.4rem;}

/* ── Metrics row ── */
.mc-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:.7rem;margin:1.1rem 0 1.5rem;}
.mc{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:.85rem 1rem;transition:border-color .15s;}
.mc:hover{border-color:var(--ac);}
.mc-l{font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--mu);margin-bottom:.3rem;}
.mc-v{font-family:var(--mono);font-size:1.4rem;font-weight:600;}
.mc-u{font-size:.68rem;color:var(--mu);margin-left:.12rem;}

/* ── Image cards ── */
.ig{display:grid;gap:.7rem;margin:.5rem 0;}
.ig-4{grid-template-columns:repeat(4,1fr);}
.ig-3{grid-template-columns:repeat(3,1fr);}
.ig-2{grid-template-columns:repeat(2,1fr);}
.ig-1{grid-template-columns:1fr;max-width:55%;}
.ig-auto{grid-template-columns:repeat(auto-fit,minmax(160px,1fr));}
.ic{background:var(--s1);border:1px solid var(--bd);border-radius:8px;overflow:hidden;transition:border-color .15s;}
.ic:hover{border-color:var(--ac);}
.ic-h{padding:.42rem .8rem;font-family:var(--mono);font-size:.6rem;text-transform:uppercase;letter-spacing:.1em;color:var(--mu);background:var(--s2);border-bottom:1px solid var(--bd);}
.badge{margin-left:.45rem;color:var(--ac);font-size:.58rem;background:var(--ac-dim);padding:.1rem .4rem;border-radius:3px;}
.ic-img{width:100%;display:block;}

/* ── Analysis box ── */
.analysis{background:var(--s1);border:1px solid var(--bd);border-radius:8px;padding:1rem 1.25rem;margin:1.25rem 0;font-family:var(--mono);font-size:.78rem;color:var(--mu);line-height:1.7;}
.analysis b{color:var(--tx);}
.analysis .hi{color:var(--ac);}

/* ── Note / warning ── */
.note{background:rgba(227,136,92,.08);border:1px solid var(--org);border-radius:6px;padding:.6rem .9rem;color:var(--org);font-size:.79rem;margin:.5rem 0;font-family:var(--mono);}

/* ── Separator ── */
.sep{border:none;border-top:1px solid var(--bd);margin:1.25rem 0;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:1px solid var(--bd);gap:.15rem;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--mu)!important;font-family:var(--mono);font-size:.77rem;padding:.4rem 1.1rem;border-radius:6px 6px 0 0;}
.stTabs [aria-selected="true"]{background:var(--s1)!important;color:var(--tx)!important;border:1px solid var(--bd)!important;border-bottom:1px solid var(--s1)!important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.1rem;}

/* ── Input widgets ── */
.stRadio>div{gap:.5rem;}
.stRadio label span{color:var(--tx)!important;font-size:.85rem!important;}
[data-testid="stFileUploader"]{border:1px dashed var(--bd)!important;border-radius:8px!important;background:var(--s1)!important;}
[data-testid="stCameraInputButton"]{background:var(--ac)!important;border:none!important;}

/* ── Empty state ── */
.empty{text-align:center;padding:3.5rem 1rem;color:var(--mu);}
.empty .icon{font-size:2.5rem;margin-bottom:.75rem;}
.empty p{font-family:var(--mono);font-size:.82rem;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ev-hdr">
  <div class="ev-eye">Pengolahan Citra Digital &nbsp;/&nbsp; Studi Kasus</div>
  <div class="ev-title">Edge<em>Vision</em></div>
  <div class="ev-sub">Deteksi tepi wajah manusia — Operator Sobel &amp; Algoritma Canny</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar — Parameter algoritma
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter")
    st.markdown("---")
    st.markdown("**Operator Sobel**")
    sobel_ksize = st.select_slider(
        "Ukuran kernel",
        options=[1, 3, 5, 7],
        value=3,
        help="Kernel lebih besar → kontur tebal, noise lebih halus",
    )
    st.markdown("---")
    st.markdown("**Algoritma Canny**")
    canny_low  = st.slider("Threshold bawah", 0, 200,  50, step=5)
    canny_high = st.slider("Threshold atas",  0, 400, 150, step=10)
    st.markdown("---")
    st.markdown("""
Kernel **1** → gradien mentah  
Kernel **3** → detail halus *(default)*  
Kernel **5** → kontur lebih tebal  
Kernel **7** → hanya tepi dominan  

Threshold Canny rendah → lebih banyak tepi
""")
    st.markdown("---")
    st.markdown("""
**Metode deteksi otomatis**  
Wajah : Haar Cascade frontal  
Mata : Region atas 55% wajah  
Mulut : Region bawah 40% wajah
""")


# ─────────────────────────────────────────────────────────────────────────────
#  Input — di atas tab supaya kedua tab pakai gambar yang sama
# ─────────────────────────────────────────────────────────────────────────────
mode = st.radio(
    "Sumber",
    ["📁  Upload dari file / galeri", "📷  Ambil foto langsung (kamera)"],
    horizontal=True,
    label_visibility="collapsed",
)

img_bgr = None

if "Upload" in mode:
    up = st.file_uploader(
        "Upload foto wajah",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    if up is not None:
        img_bgr = load_image(up.read())
        if img_bgr is None:
            st.error("❌ File tidak dapat dibaca — pastikan format dan file tidak rusak.")
else:
    cam = st.camera_input("Ambil foto", label_visibility="collapsed")
    if cam is not None:
        img_bgr = load_image(cam.read())

st.markdown('<hr class="sep"/>', unsafe_allow_html=True)


# ── Placeholder kalau belum ada gambar ──────────────────────────────────────
if img_bgr is None:
    st.markdown("""
    <div class="empty">
      <div class="icon">🖼️</div>
      <p>Upload foto wajah atau aktifkan kamera untuk memulai analisis tepi</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  Proses gambar (setelah ada input)
# ─────────────────────────────────────────────────────────────────────────────
gray_arr             = to_gray(img_bgr)
gray_rgb             = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2RGB)
sobel_img, sobel_ms  = apply_sobel(gray_arr, ksize=sobel_ksize)
canny_img, canny_ms  = apply_canny(gray_arr, low=canny_low, high=canny_high)
sobel_dens           = edge_density(sobel_img)
canny_dens           = edge_density(canny_img)
h_px, w_px           = img_bgr.shape[:2]


# ─────────────────────────────────────────────────────────────────────────────
#  Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_edge, tab_face = st.tabs(["🔬  Deteksi Tepi", "👤  Deteksi Fitur Wajah"])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — Deteksi Tepi
# ══════════════════════════════════════════════════════════════════════════════
with tab_edge:

    # Metric cards
    row = (
        '<div class="mc-row">'
        + metric("Resolusi",          f"{w_px}×{h_px}", "px")
        + metric("Sobel — Waktu",     f"{sobel_ms:.1f}", "ms")
        + metric("Canny — Waktu",     f"{canny_ms:.1f}", "ms")
        + metric("Sobel — Dens. Tepi",f"{sobel_dens:.1f}", "%", hi=True)
        + metric("Canny — Dens. Tepi",f"{canny_dens:.1f}", "%", hi=True)
        + '</div>'
    )
    st.markdown(row, unsafe_allow_html=True)

    # 4 image cards
    grid = (
        '<div class="ig ig-4">'
        + img_card(to_rgb(img_bgr), "Original",   "RGB")
        + img_card(gray_rgb,         "Grayscale")
        + img_card(sobel_img,         "Sobel",     f"kernel {sobel_ksize}×{sobel_ksize}")
        + img_card(canny_img,         "Canny",     f"T [{canny_low}, {canny_high}]")
        + '</div>'
    )
    st.markdown(grid, unsafe_allow_html=True)

    # Interpretasi otomatis (bahan Bab 4)
    faster   = "Canny" if canny_ms < sobel_ms else "Sobel"
    leaner   = "Canny" if canny_dens < sobel_dens else "Sobel"
    ratio_sc = sobel_dens / canny_dens if canny_dens > 0 else float("inf")

    st.markdown(f"""
    <div class="analysis">
    ▸ Sobel menghasilkan <b>{sobel_dens:.1f}%</b> piksel tepi dalam
      <b>{sobel_ms:.1f} ms</b> — gradien kontinu (abu-abu), sensitif terhadap noise.<br>
    ▸ Canny menghasilkan <b>{canny_dens:.1f}%</b> piksel tepi dalam
      <b>{canny_ms:.1f} ms</b> — tepi biner (hitam-putih), presisi tinggi
      berkat <em>non-maximum suppression</em> + <em>hysteresis thresholding</em>.<br>
    ▸ Kepadatan Sobel ÷ Canny = <b>{ratio_sc:.2f}×</b> — Sobel lebih "berisik";
      Canny lebih selektif mempertahankan tepi yang signifikan secara perceptual.<br>
    ▸ Algoritma lebih cepat pada foto ini: <span class="hi"><b>{faster}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Perbandingan lanjutan dengan parameter berbeda (bahan Bab 4.3)
    with st.expander("📊  Perbandingan multi-parameter (bahan data Bab 4.3)"):
        configs = [
            ("Sobel k=1",  *apply_sobel(gray_arr, ksize=1)),
            ("Sobel k=3",  *apply_sobel(gray_arr, ksize=3)),
            ("Sobel k=5",  *apply_sobel(gray_arr, ksize=5)),
            ("Sobel k=7",  *apply_sobel(gray_arr, ksize=7)),
            ("Canny T=30/90",  *apply_canny(gray_arr, 30,  90)),
            ("Canny T=50/150", *apply_canny(gray_arr, 50, 150)),
            ("Canny T=80/200", *apply_canny(gray_arr, 80, 200)),
        ]

        rows_html = ""
        for label, img_r, ms in configs:
            dens = edge_density(img_r)
            rows_html += (
                f'<tr>'
                f'<td style="padding:.35rem .7rem;border-bottom:1px solid var(--bd);'
                f'color:var(--tx)">{label}</td>'
                f'<td style="padding:.35rem .7rem;border-bottom:1px solid var(--bd);'
                f'color:var(--ac);text-align:right">{dens:.2f}%</td>'
                f'<td style="padding:.35rem .7rem;border-bottom:1px solid var(--bd);'
                f'color:var(--mu);text-align:right">{ms:.2f} ms</td>'
                f'</tr>'
            )

        table = (
            '<table style="width:100%;border-collapse:collapse;font-family:var(--mono);'
            'font-size:.78rem;margin-top:.5rem;">'
            '<thead><tr>'
            '<th style="padding:.35rem .7rem;border-bottom:1px solid var(--bd);'
            'color:var(--mu);text-align:left">Konfigurasi</th>'
            '<th style="padding:.35rem .7rem;border-bottom:1px solid var(--bd);'
            'color:var(--mu);text-align:right">Dens. Tepi</th>'
            '<th style="padding:.35rem .7rem;border-bottom:1px solid var(--bd);'
            'color:var(--mu);text-align:right">Waktu</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            '</table>'
        )
        st.markdown(table, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Deteksi Fitur Wajah
# ══════════════════════════════════════════════════════════════════════════════
with tab_face:
    with st.spinner("Mendeteksi fitur wajah…"):
        det = detect_face(img_bgr)

    for note in det["notes"]:
        st.markdown(f'<div class="note">⚠ {note}</div>', unsafe_allow_html=True)

    # ── Baris 1: Annotated + Face crop ──────────────────────────────────────
    parts1 = [img_card(det["annotated"], "Anotasi", "bounding box")]
    if det["face"] is not None:
        parts1.append(img_card(det["face"], "Wajah", "crop"))

    cls1 = "ig-2" if len(parts1) == 2 else "ig-1"
    st.markdown(f'<div class="ig {cls1}">{"".join(parts1)}</div>',
                unsafe_allow_html=True)

    # ── Baris 2: Mata + Mulut ────────────────────────────────────────────────
    parts2 = []
    if det.get("right_eye") is not None:
        parts2.append(img_card(det["right_eye"], "Mata Kanan", "crop"))
    if det.get("left_eye") is not None:
        parts2.append(img_card(det["left_eye"],  "Mata Kiri",  "crop"))
    if det.get("mouth") is not None:
        parts2.append(img_card(det["mouth"],     "Area Mulut", "crop"))

    if parts2:
        n    = len(parts2)
        cls2 = ["", "ig-1", "ig-2", "ig-3", "ig-3"][min(n, 4)]
        st.markdown(f'<div class="ig {cls2}">{"".join(parts2)}</div>',
                    unsafe_allow_html=True)
    elif det["face"] is not None:
        st.markdown('<div class="note">⚠ Mata dan mulut tidak dapat dideteksi pada foto ini.</div>',
                    unsafe_allow_html=True)

    # ── Deteksi tepi pada crop fitur wajah ──────────────────────────────────
    if det["face"] is not None:
        with st.expander("🔬  Deteksi tepi pada fitur wajah (Sobel & Canny)"):
            targets = [
                ("Wajah",      det["face"]),
                ("Mata Kanan", det.get("right_eye")),
                ("Mata Kiri",  det.get("left_eye")),
                ("Area Mulut", det.get("mouth")),
            ]
            for name, crop_rgb in targets:
                if crop_rgb is None:
                    continue
                crop_bgr  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                crop_gray = to_gray(crop_bgr)
                s_img, _  = apply_sobel(crop_gray, ksize=sobel_ksize)
                c_img, _  = apply_canny(crop_gray, low=canny_low, high=canny_high)

                st.markdown(f"**{name}**")
                row = (
                    '<div class="ig ig-3">'
                    + img_card(crop_rgb, f"{name} — Original", "RGB")
                    + img_card(s_img,    f"{name} — Sobel",    f"k={sobel_ksize}")
                    + img_card(c_img,    f"{name} — Canny",    f"T[{canny_low},{canny_high}]")
                    + '</div>'
                )
                st.markdown(row, unsafe_allow_html=True)
                st.markdown("")

    # ── Catatan metodologis ──────────────────────────────────────────────────
    st.markdown("""
    <div class="analysis" style="margin-top:1.25rem;">
    ▸ <b>Wajah</b> : Haar Cascade frontal — scaleFactor 1.1, minNeighbors 5, minSize 80×80 px<br>
    ▸ <b>Mata</b> : Haar Eye Cascade — dibatasi pada 55% area atas wajah untuk
      menekan false positive dari hidung/alis<br>
    ▸ <b>Mulut</b> : Haar Smile Cascade — dibatasi pada 40% area bawah wajah;
      estimasi proporsional digunakan sebagai fallback jika senyuman tidak terdeteksi
    </div>
    """, unsafe_allow_html=True)
