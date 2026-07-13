"""app.py — EdgeVision v3 | Deteksi Tepi Wajah Manusia"""
import io, base64, zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2, numpy as np
import streamlit as st
from PIL import Image
from imgproc import (
    load_image, to_rgb, to_gray,
    apply_sobel, apply_canny, edge_density, detect_face,
)

st.set_page_config(
    page_title="EdgeVision — Deteksi Tepi Wajah",
    page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palet warna (langsung hex, tidak pakai CSS variable) ────────────────────
BG  = "#080C12"; S1 = "#0F1419"; S2 = "#161C24"; S3 = "#1E2631"
BD  = "#2A3340"; AC = "#4D9EFF"; G  = "#4ADE80";  O  = "#FB923C"
TX  = "#E8EFF8"; MU = "#6B7A8D"

# ── CSS: hanya untuk override widget Streamlit ──────────────────────────────
st.markdown(f"""<style>
.stApp{{background:{BG}!important;}}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]{{display:none!important;}}
.stApp>header{{background:transparent!important;}}
p,li{{color:{TX};}}
[data-testid="stSidebar"]{{background:{S1}!important;border-right:1px solid {BD}!important;}}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] li{{color:{MU}!important;font-size:.78rem;}}
[data-testid="stSidebar"] h3{{color:{TX}!important;font-size:.9rem;font-weight:600;}}
[data-testid="stSidebar"] label{{color:{MU}!important;font-size:.8rem!important;}}
[data-testid="stSidebar"] hr{{border-color:{BD}!important;}}
[data-testid="stSidebar"] strong{{color:{TX}!important;}}
.stTabs [data-baseweb="tab-list"]{{background:transparent!important;border-bottom:1px solid {BD};gap:.1rem;}}
.stTabs [data-baseweb="tab"]{{background:transparent!important;color:{MU}!important;font-family:monospace;font-size:.76rem;padding:.4rem 1.1rem;border-radius:6px 6px 0 0;}}
.stTabs [aria-selected="true"]{{background:{S2}!important;color:{TX}!important;border:1px solid {BD}!important;border-bottom:1px solid {S2}!important;}}
.stTabs [data-baseweb="tab-panel"]{{padding-top:1rem;}}
[data-testid="stFileUploader"]{{border:1px dashed {BD}!important;border-radius:10px!important;background:{S2}!important;}}
.stRadio label span{{color:{TX}!important;font-size:.85rem!important;}}
[data-testid="stDownloadButton"] button{{background:{AC}!important;border:none!important;color:#000!important;font-weight:700!important;border-radius:8px!important;}}
.stCameraInput video{{border-radius:8px!important;border:1px solid {BD}!important;}}
[data-testid="stExpander"]{{background:{S2}!important;border:1px solid {BD}!important;border-radius:8px!important;}}
</style>""", unsafe_allow_html=True)

# ── Helper: encode array RGB → base64 ───────────────────────────────────────
def enc(arr: np.ndarray, q: int = 88) -> tuple:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    try:
        pil.save(buf, format="WEBP", quality=q); mime = "image/webp"
    except Exception:
        buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=q); mime = "image/jpeg"
    return base64.b64encode(buf.getvalue()).decode(), mime

# ── Helper: satu image card (inline styles, zero CSS-var dependency) ─────────
def card(arr: np.ndarray, judul: str, sub: str = "") -> str:
    b64, mime = enc(arr)
    badge = (f'<span style="margin-left:.45rem;color:{AC};font-size:.58rem;'
             f'background:rgba(77,158,255,.1);padding:.08rem .4rem;border-radius:4px;'
             f'border:1px solid rgba(77,158,255,.2);">{sub}</span>') if sub else ""
    return (f'<div style="background:{S2};border:1px solid {BD};border-radius:10px;overflow:hidden;">'
            f'<div style="padding:.42rem .8rem;font-family:monospace;font-size:.59rem;'
            f'text-transform:uppercase;letter-spacing:.1em;color:{MU};background:{S3};'
            f'border-bottom:1px solid {BD};display:flex;align-items:center;">'
            f'{judul}{badge}</div>'
            f'<img src="data:{mime};base64,{b64}" style="width:100%;display:block;"/>'
            f'</div>')

# ── Helper: grid n kolom ─────────────────────────────────────────────────────
def grid(*cards, cols: int = 4) -> str:
    return (f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
            f'gap:.65rem;margin:.5rem 0;">' + "".join(cards) + '</div>')

# ── Helper: metric card (inline) ─────────────────────────────────────────────
def met(label: str, val: str, unit: str = "", hi: bool = False) -> str:
    c = AC if hi else TX
    return (f'<div style="background:{S2};border:1px solid {BD};border-radius:10px;padding:.9rem 1rem;">'
            f'<div style="font-family:monospace;font-size:.58rem;text-transform:uppercase;'
            f'letter-spacing:.12em;color:{MU};margin-bottom:.35rem;">{label}</div>'
            f'<div style="font-family:monospace;font-size:1.45rem;font-weight:700;color:{c};">'
            f'{val}<span style="font-size:.64rem;color:{MU};margin-left:.1rem;">{unit}</span></div></div>')

def met_row(*items) -> str:
    return ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));'
            f'gap:.65rem;margin:1rem 0 1.5rem;">' + "".join(met(*i) for i in items) + '</div>')

# ── Helper: kotak analisis / notif ───────────────────────────────────────────
def kotak(html: str, border_color: str = AC) -> str:
    return (f'<div style="background:{S2};border:1px solid {BD};border-left:3px solid {border_color};'
            f'border-radius:0 8px 8px 0;padding:1rem 1.25rem;margin:1.2rem 0;'
            f'font-size:.82rem;color:{MU};line-height:1.8;">{html}</div>')

def notif_ok(t: str) -> str:
    return (f'<div style="background:rgba(74,222,128,.07);border:1px solid rgba(74,222,128,.3);'
            f'border-radius:8px;padding:.62rem 1rem;color:{G};font-size:.8rem;'
            f'margin:.5rem 0;font-family:monospace;">{t}</div>')

def notif_warn(t: str) -> str:
    return (f'<div style="background:rgba(251,146,60,.07);border:1px solid rgba(251,146,60,.3);'
            f'border-radius:8px;padding:.62rem 1rem;color:{O};font-size:.8rem;'
            f'margin:.5rem 0;font-family:monospace;">{t}</div>')

def sep() -> str:
    return f'<hr style="border:none;border-top:1px solid {BD};margin:1.25rem 0;"/>'

# ── Helper: grafik matplotlib ────────────────────────────────────────────────
def buat_grafik(gray_arr: np.ndarray):
    cfgs = [
        ("Sobel\nk=1",    *apply_sobel(gray_arr, 1)),
        ("Sobel\nk=3",    *apply_sobel(gray_arr, 3)),
        ("Sobel\nk=5",    *apply_sobel(gray_arr, 5)),
        ("Sobel\nk=7",    *apply_sobel(gray_arr, 7)),
        ("Canny\n30/90",  *apply_canny(gray_arr, 30, 90)),
        ("Canny\n50/150", *apply_canny(gray_arr, 50, 150)),
        ("Canny\n80/200", *apply_canny(gray_arr, 80, 200)),
    ]
    labels = [c[0] for c in cfgs]
    dens   = [edge_density(c[1]) for c in cfgs]
    times  = [c[2] for c in cfgs]
    clrs   = [AC if "Sobel" in l else G for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.8))
    fig.patch.set_facecolor(S1)

    for ax in (ax1, ax2):
        ax.set_facecolor(S2)
        ax.tick_params(colors=MU, labelsize=7.5)
        for sp in ("bottom", "left"):  ax.spines[sp].set_color(BD)
        for sp in ("top", "right"):    ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=BD, linewidth=.5, alpha=.6)
        ax.set_axisbelow(True)

    bars1 = ax1.bar(labels, dens, color=clrs, width=.6, alpha=.88, zorder=2)
    ax1.set_title("Kepadatan Tepi (%)", color=TX, fontsize=10, pad=8, fontweight="bold")
    ax1.set_ylabel("%", color=MU, fontsize=8)
    for bar, v in zip(bars1, dens):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + .3,
                 f"{v:.1f}%", ha="center", va="bottom", color=TX, fontsize=7, fontweight="bold")

    bars2 = ax2.bar(labels, times, color=clrs, width=.6, alpha=.88, zorder=2)
    ax2.set_title("Waktu Komputasi (ms)", color=TX, fontsize=10, pad=8, fontweight="bold")
    ax2.set_ylabel("ms", color=MU, fontsize=8)
    for bar, v in zip(bars2, times):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + .05,
                 f"{v:.1f}", ha="center", va="bottom", color=TX, fontsize=7, fontweight="bold")

    legend = [mpatches.Patch(color=AC, label="Operator Sobel"),
              mpatches.Patch(color=G,  label="Algoritma Canny")]
    ax1.legend(handles=legend, facecolor=S3, edgecolor=BD, labelcolor=TX, fontsize=8)
    plt.tight_layout(pad=1.5)
    return fig

# ── Helper: ZIP semua hasil ──────────────────────────────────────────────────
def buat_zip(img_bgr, gray_rgb, sob_img, sob_heat, can_img) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for nama, arr in [("01_original.jpg",  to_rgb(img_bgr)),
                          ("02_grayscale.jpg",  gray_rgb),
                          ("03_sobel.jpg",      sob_img),
                          ("04_sobel_heatmap.jpg", sob_heat),
                          ("05_canny.jpg",      can_img)]:
            ib = io.BytesIO()
            Image.fromarray(arr.astype(np.uint8)).save(ib, "JPEG", quality=92)
            zf.writestr(nama, ib.getvalue())
    return buf.getvalue()

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER  (inline styles agar tidak terpengaruh CSS-var)
# ═══════════════════════════════════════════════════════════════════════════════
pills = ["Python 3 + OpenCV", "Haar Cascade Classifier",
         "Non-max Suppression", "Hysteresis Thresholding"]
pill_html = "".join(
    f'<span style="background:{S3};border:1px solid {BD};color:{MU};border-radius:6px;'
    f'padding:.2rem .65rem;font-size:.67rem;font-family:monospace;">{p}</span>'
    for p in pills
)
st.markdown(f"""
<div style="padding:1.75rem 0 1.4rem;border-bottom:1px solid {BD};margin-bottom:1.75rem;">
  <div style="display:inline-flex;align-items:center;gap:.4rem;background:rgba(77,158,255,.08);
    border:1px solid rgba(77,158,255,.22);color:{AC};border-radius:20px;padding:.2rem .75rem;
    font-size:.62rem;font-family:monospace;letter-spacing:.16em;text-transform:uppercase;
    margin-bottom:.85rem;">
    Pengolahan Citra Digital &nbsp;/&nbsp; Studi Kasus
  </div>
  <div style="font-size:2.25rem;font-weight:800;letter-spacing:-.04em;color:{TX};
    line-height:1.05;margin:0;">
    Edge<span style="color:{AC};">Vision</span>
  </div>
  <div style="color:{MU};font-size:.88rem;margin-top:.5rem;">
    Deteksi tepi wajah manusia menggunakan Operator Sobel &amp; Algoritma Canny
  </div>
  <div style="display:flex;gap:.5rem;margin-top:.8rem;flex-wrap:wrap;">{pill_html}</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Parameter")
    st.markdown("---")
    st.markdown("**Operator Sobel**")
    sobel_k = st.select_slider("Ukuran kernel", options=[1, 3, 5, 7], value=3,
                               help="Kernel lebih besar → kontur tebal, noise halus")
    st.markdown("---")
    st.markdown("**Algoritma Canny**")
    c_lo = st.slider("Threshold bawah", 0, 200,  50, step=5)
    c_hi = st.slider("Threshold atas",  0, 400, 150, step=10)
    st.markdown("---")
    st.markdown("""**Panduan kernel Sobel:**  
`1` → gradien mentah  
`3` → detail halus *(default)*  
`5` → kontur lebih tebal  
`7` → hanya tepi dominan  

**Panduan threshold Canny:**  
Nilai rendah → lebih banyak tepi terdeteksi  
Nilai tinggi → hanya tepi yang kuat""")
    st.markdown("---")
    st.markdown("""**Metode deteksi otomatis:**  
Wajah: Haar Cascade frontal  
Mata: Region atas 55% wajah  
Mulut: Region bawah 40% wajah""")

# ═══════════════════════════════════════════════════════════════════════════════
#  INPUT
# ═══════════════════════════════════════════════════════════════════════════════
mode = st.radio("mode", ["📁  Upload dari file / galeri", "📷  Kamera langsung"],
                horizontal=True, label_visibility="collapsed")

img_bgr = None
if "Upload" in mode:
    up = st.file_uploader("Upload foto wajah (JPG / PNG / WEBP)",
                          type=["jpg","jpeg","png","bmp","webp"],
                          label_visibility="collapsed")
    if up:
        img_bgr = load_image(up.read())
        if img_bgr is None:
            st.error("❌ File tidak dapat dibaca. Coba format JPG atau PNG.")
else:
    st.markdown(
        f'<div style="font-size:.75rem;color:{MU};font-family:monospace;'
        f'background:{S2};border:1px solid {BD};border-radius:8px;'
        f'padding:.55rem .9rem;margin-bottom:.5rem;">'
        f'ⓘ Jika tampilan kamera tidak muncul setelah mengklik <b style="color:{TX}">Allow</b>, '
        f'<b style="color:{AC}">reload halaman ini sekali</b> — izin kamera baru aktif setelah reload.</div>',
        unsafe_allow_html=True
    )
    cam = st.camera_input("Ambil foto", label_visibility="collapsed")
    if cam:
        img_bgr = load_image(cam.read())

st.markdown(sep(), unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────────────────────────
if img_bgr is None:
    st.markdown(f"""
    <div style="text-align:center;padding:4rem 1rem;color:{MU};">
      <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
      <div style="color:{TX};font-weight:600;font-size:1.1rem;margin:.5rem 0;">Siap menganalisis</div>
      <div style="font-family:monospace;font-size:.8rem;">
        Upload foto wajah atau aktifkan kamera untuk memulai deteksi tepi</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Proses gambar ─────────────────────────────────────────────────────────────
gray_arr           = to_gray(img_bgr)
gray_rgb           = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2RGB)
sobel_img, sob_ms  = apply_sobel(gray_arr, ksize=sobel_k)
canny_img, can_ms  = apply_canny(gray_arr, low=c_lo,   high=c_hi)
sob_dens           = edge_density(sobel_img)
can_dens           = edge_density(canny_img)
h_px, w_px         = img_bgr.shape[:2]
ratio              = sob_dens / can_dens if can_dens > 0 else 0

# Heatmap plasma Sobel
sob_gray = cv2.cvtColor(sobel_img, cv2.COLOR_RGB2GRAY)
sob_heat = cv2.cvtColor(cv2.applyColorMap(sob_gray, cv2.COLORMAP_PLASMA), cv2.COLOR_BGR2RGB)

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════
t1, t2 = st.tabs(["🔬  Deteksi Tepi", "👤  Deteksi Fitur Wajah"])

# ───────────────────────────────────────────────────────────────────────────────
#  TAB 1 — DETEKSI TEPI
# ───────────────────────────────────────────────────────────────────────────────
with t1:
    # Metric row
    lebih_cepat = "Canny" if can_ms < sob_ms else "Sobel"
    st.markdown(met_row(
        ("Resolusi",           f"{w_px}×{h_px}", "px",  False),
        ("Waktu — Sobel",      f"{sob_ms:.1f}",  "ms",  False),
        ("Waktu — Canny",      f"{can_ms:.1f}",  "ms",  False),
        ("Kepadatan Sobel",    f"{sob_dens:.1f}", "%",   True),
        ("Kepadatan Canny",    f"{can_dens:.1f}", "%",   True),
        ("Rasio Sobel / Canny",f"{ratio:.1f}",   "×",   True),
    ), unsafe_allow_html=True)

    # Toggle tampilan Sobel: Grayscale vs Heatmap Plasma
    viz = st.radio("Tampilan Sobel",
                   ["Grayscale (standar)", "Heatmap Plasma (visualisasi warna)"],
                   horizontal=True, label_visibility="collapsed")
    use_heat = "Plasma" in viz
    sob_disp  = sob_heat  if use_heat else sobel_img
    sob_badge = "COLORMAP_PLASMA" if use_heat else f"Kernel {sobel_k}×{sobel_k}"

    # 4 image cards
    st.markdown(grid(
        card(to_rgb(img_bgr), "Original",   "RGB"),
        card(gray_rgb,         "Grayscale", "BGR → Gray"),
        card(sob_disp,         "Sobel",     sob_badge),
        card(canny_img,        "Canny",     f"T [{c_lo}, {c_hi}]"),
    ), unsafe_allow_html=True)

    # Analisis otomatis
    st.markdown(kotak(
        f'▸ <b style="color:{TX}">Sobel</b> — '
        f'<b style="color:{TX}">{sob_dens:.1f}%</b> piksel terdeteksi sebagai tepi dalam '
        f'<b style="color:{TX}">{sob_ms:.1f} ms</b>. '
        f'Peta gradien kontinu (nilai abu-abu) dihitung melalui konvolusi kernel sumbu-X dan sumbu-Y.<br>'
        f'▸ <b style="color:{TX}">Canny</b> — '
        f'<b style="color:{TX}">{can_dens:.1f}%</b> piksel terdeteksi sebagai tepi dalam '
        f'<b style="color:{TX}">{can_ms:.1f} ms</b>. '
        f'Tepi biner presisi tinggi melalui 4 tahap: '
        f'<em>Gaussian smoothing → Gradien Sobel → Non-maximum suppression → Hysteresis thresholding</em>.<br>'
        f'▸ Rasio kepadatan Sobel ÷ Canny = <span style="color:{AC};font-weight:700;">{ratio:.2f}×</span> — '
        f'Sobel lebih "berisik" karena tidak menipis tepi. '
        f'Canny selektif mempertahankan tepi yang signifikan.<br>'
        f'▸ Algoritma lebih cepat pada foto ini: '
        f'<span style="color:{G};font-weight:700;">{lebih_cepat}</span>'
    ), unsafe_allow_html=True)

    # Grafik perbandingan
    with st.expander("📊  Grafik perbandingan semua metode & parameter"):
        fig = buat_grafik(gray_arr)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Tabel perbandingan
    with st.expander("📋  Tabel perbandingan parameter"):
        cfgs_tbl = [
            ("Sobel k=1",      *apply_sobel(gray_arr, 1)),
            ("Sobel k=3",      *apply_sobel(gray_arr, 3)),
            ("Sobel k=5",      *apply_sobel(gray_arr, 5)),
            ("Sobel k=7",      *apply_sobel(gray_arr, 7)),
            ("Canny T=30/90",  *apply_canny(gray_arr, 30,  90)),
            ("Canny T=50/150", *apply_canny(gray_arr, 50, 150)),
            ("Canny T=80/200", *apply_canny(gray_arr, 80, 200)),
        ]
        baris = ""
        for nama, img_r, ms in cfgs_tbl:
            d = edge_density(img_r)
            baris += (f'<tr style="transition:background .15s;" '
                      f'onmouseover="this.style.background=\'{S3}\'" '
                      f'onmouseout="this.style.background=\'transparent\'">'
                      f'<td style="padding:.42rem .85rem;color:{TX};border-bottom:1px solid {BD};">{nama}</td>'
                      f'<td style="padding:.42rem .85rem;color:{AC};text-align:right;border-bottom:1px solid {BD};font-family:monospace;">{d:.2f}%</td>'
                      f'<td style="padding:.42rem .85rem;color:{MU};text-align:right;border-bottom:1px solid {BD};font-family:monospace;">{ms:.2f} ms</td>'
                      f'</tr>')
        st.markdown(
            f'<table style="width:100%;border-collapse:collapse;font-family:monospace;font-size:.8rem;">'
            f'<thead><tr style="border-bottom:2px solid {BD};">'
            f'<th style="padding:.42rem .85rem;color:{MU};text-align:left;font-size:.6rem;'
            f'text-transform:uppercase;letter-spacing:.1em;">Konfigurasi</th>'
            f'<th style="padding:.42rem .85rem;color:{MU};text-align:right;font-size:.6rem;'
            f'text-transform:uppercase;letter-spacing:.1em;">Kepadatan Tepi</th>'
            f'<th style="padding:.42rem .85rem;color:{MU};text-align:right;font-size:.6rem;'
            f'text-transform:uppercase;letter-spacing:.1em;">Waktu Komputasi</th>'
            f'</tr></thead><tbody>{baris}</tbody></table>',
            unsafe_allow_html=True
        )

    # Download
    st.markdown("")
    zip_bytes = buat_zip(img_bgr, gray_rgb, sobel_img, sob_heat, canny_img)
    st.download_button(
        "📥  Unduh semua hasil sebagai ZIP",
        data=zip_bytes,
        file_name="edgevision_hasil.zip",
        mime="application/zip",
    )

# ───────────────────────────────────────────────────────────────────────────────
#  TAB 2 — DETEKSI FITUR WAJAH
# ───────────────────────────────────────────────────────────────────────────────
with t2:
    with st.spinner("Mendeteksi fitur wajah..."):
        det = detect_face(img_bgr)

    if det["notes"]:
        for n in det["notes"]:
            st.markdown(notif_warn(f"⚠ {n}"), unsafe_allow_html=True)
    else:
        st.markdown(notif_ok("✓ Wajah terdeteksi — fitur berhasil diisolasi"),
                    unsafe_allow_html=True)

    # Baris 1: anotasi + wajah crop
    b1 = [card(det["annotated"], "Anotasi Bounding Box", "Haar Cascade")]
    if det["face"]:
        b1.append(card(det["face"], "Crop Wajah", "Region terbesar"))
    st.markdown(grid(*b1, cols=len(b1)), unsafe_allow_html=True)

    # Baris 2: mata + mulut
    b2 = []
    if det.get("right_eye"): b2.append(card(det["right_eye"], "Mata Kanan", "crop"))
    if det.get("left_eye"):  b2.append(card(det["left_eye"],  "Mata Kiri",  "crop"))
    if det.get("mouth"):     b2.append(card(det["mouth"],     "Area Mulut", "crop"))
    if b2:
        st.markdown(grid(*b2, cols=min(len(b2), 3)), unsafe_allow_html=True)
    elif det["face"] is not None:
        st.markdown(notif_warn("⚠ Mata dan mulut tidak dapat dideteksi pada foto ini."),
                    unsafe_allow_html=True)

    # Deteksi tepi pada setiap fitur
    if det["face"]:
        with st.expander("🔬  Deteksi tepi Sobel & Canny pada setiap fitur wajah"):
            for nama, crop_rgb in [("Wajah",      det["face"]),
                                   ("Mata Kanan",  det.get("right_eye")),
                                   ("Mata Kiri",   det.get("left_eye")),
                                   ("Area Mulut",  det.get("mouth"))]:
                if crop_rgb is None: continue
                bgr_c = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                gr_c  = to_gray(bgr_c)
                si, _ = apply_sobel(gr_c, ksize=sobel_k)
                ci, _ = apply_canny(gr_c, low=c_lo, high=c_hi)
                sd, cd = edge_density(si), edge_density(ci)
                st.markdown(
                    f'<div style="font-family:monospace;font-size:.72rem;color:{MU};margin:.75rem 0 .3rem;">'
                    f'{nama} &nbsp;—&nbsp; Sobel: <b style="color:{TX}">{sd:.1f}%</b>'
                    f' &nbsp;|&nbsp; Canny: <b style="color:{AC}">{cd:.1f}%</b></div>',
                    unsafe_allow_html=True
                )
                st.markdown(grid(
                    card(crop_rgb, f"{nama} — Asli",  "RGB"),
                    card(si,       f"{nama} — Sobel", f"k={sobel_k}"),
                    card(ci,       f"{nama} — Canny", f"T[{c_lo},{c_hi}]"),
                    cols=3
                ), unsafe_allow_html=True)

    # Catatan metodologis
    st.markdown(kotak(
        f'▸ <b style="color:{TX}">Deteksi wajah</b>: Haar Cascade frontal — '
        f'scaleFactor 1.1, minNeighbors 5, minSize 80×80 px<br>'
        f'▸ <b style="color:{TX}">Deteksi mata</b>: Haar Eye Cascade — '
        f'dibatasi pada 55% area atas wajah untuk menekan <em>false positive</em> dari hidung/alis<br>'
        f'▸ <b style="color:{TX}">Deteksi mulut</b>: Haar Smile Cascade — '
        f'dibatasi pada 40% area bawah wajah; estimasi geometri proporsional sebagai '
        f'<em>fallback</em> jika senyuman tidak terdeteksi'
    ), unsafe_allow_html=True)
