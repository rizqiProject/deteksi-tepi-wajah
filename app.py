"""app.py — EdgeVision: Deteksi Tepi Wajah | Pengolahan Citra Digital"""

import io, base64, cv2, numpy as np
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

# ─── Encode gambar → base64 WEBP ────────────────────────────────────────────
def enc(arr: np.ndarray, q: int = 88) -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="WEBP", quality=q)
    return base64.b64encode(buf.getvalue()).decode()

def card(arr: np.ndarray, judul: str, sub: str = "") -> str:
    b = enc(arr)
    s = f'<span class="badge">{sub}</span>' if sub else ""
    return (f'<div class="ic"><div class="ic-h">{judul}{s}</div>'
            f'<img src="data:image/webp;base64,{b}" class="ic-img"/></div>')

def met(label: str, nilai: str, satuan: str = "", biru: bool = False) -> str:
    warna = "#4D9EFF" if biru else "#E6EDF3"
    return (f'<div class="mc"><div class="mc-l">{label}</div>'
            f'<div class="mc-v" style="color:{warna}">{nilai}'
            f'<span class="mc-u">{satuan}</span></div></div>')

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
:root{
  --bg:#080C12; --s1:#0F1419; --s2:#161C24; --s3:#1E2631;
  --bd:#2A3340; --ac:#4D9EFF; --ac2:#7BC4FF;
  --grn:#4ADE80; --org:#FB923C; --red:#F87171; --pur:#A78BFA;
  --tx:#E8EFF8; --mu:#6B7A8D; --mu2:#4A5568;
  --font: -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  --mono: 'SF Mono','Fira Code','Cascadia Code',monospace;
}
.stApp{background:var(--bg)!important;font-family:var(--font);color:var(--tx);}
#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"]
{display:none!important;}
.stApp>header{background:transparent!important;}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:var(--s1)!important;
  border-right:1px solid var(--bd)!important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li{color:var(--mu)!important;font-size:.78rem;}
[data-testid="stSidebar"] h3{color:var(--tx)!important;font-size:.9rem;font-weight:600;}
[data-testid="stSidebar"] label{color:var(--mu)!important;font-size:.8rem!important;}
[data-testid="stSidebar"] hr{border-color:var(--bd)!important;}
[data-testid="stSidebar"] strong{color:var(--tx)!important;}

/* ── Header ── */
.ev-wrap{
  position:relative;
  padding:2rem 0 1.6rem;
  border-bottom:1px solid var(--bd);
  margin-bottom:1.8rem;
  overflow:hidden;
}
.ev-wrap::before{
  content:'';
  position:absolute;top:0;left:-5%;
  width:55%;height:100%;
  background:radial-gradient(ellipse at 30% 50%,rgba(77,158,255,.06) 0%,transparent 70%);
  pointer-events:none;
}
.ev-tag{
  display:inline-flex;align-items:center;gap:.4rem;
  background:rgba(77,158,255,.08);border:1px solid rgba(77,158,255,.25);
  color:var(--ac);border-radius:20px;padding:.2rem .75rem;
  font-size:.65rem;font-family:var(--mono);letter-spacing:.12em;
  text-transform:uppercase;margin-bottom:.9rem;
}
.ev-tag::before{content:'●';font-size:.45rem;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.3;}}
.ev-title{
  font-size:2.3rem;font-weight:800;letter-spacing:-.04em;
  color:var(--tx);margin:0;line-height:1.05;
}
.ev-title em{
  color:transparent;
  background:linear-gradient(135deg,var(--ac) 0%,var(--ac2) 100%);
  -webkit-background-clip:text;background-clip:text;
  font-style:normal;
}
.ev-sub{color:var(--mu);font-size:.88rem;margin-top:.5rem;letter-spacing:.01em;}
.ev-pills{display:flex;gap:.5rem;margin-top:.9rem;flex-wrap:wrap;}
.pill{
  background:var(--s2);border:1px solid var(--bd);
  color:var(--mu);border-radius:6px;padding:.2rem .65rem;
  font-size:.67rem;font-family:var(--mono);
}

/* ── Metrics ── */
.mc-row{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:.65rem;margin:1.1rem 0 1.5rem;
}
.mc{
  background:var(--s2);border:1px solid var(--bd);
  border-radius:10px;padding:.9rem 1rem;
  transition:border-color .2s,background .2s;cursor:default;
}
.mc:hover{border-color:var(--ac);background:var(--s3);}
.mc-l{
  font-family:var(--mono);font-size:.58rem;text-transform:uppercase;
  letter-spacing:.12em;color:var(--mu);margin-bottom:.35rem;
}
.mc-v{font-size:1.5rem;font-weight:700;line-height:1;}
.mc-u{font-size:.65rem;color:var(--mu);margin-left:.1rem;font-weight:400;}

/* ── Image cards ── */
.ig{display:grid;gap:.65rem;margin:.5rem 0;}
.ig-4{grid-template-columns:repeat(4,1fr);}
.ig-3{grid-template-columns:repeat(3,1fr);}
.ig-2{grid-template-columns:repeat(2,1fr);}
.ig-1{grid-template-columns:minmax(0,1fr);max-width:48%;}
.ic{
  background:var(--s2);border:1px solid var(--bd);
  border-radius:10px;overflow:hidden;
  transition:border-color .2s,box-shadow .2s;
}
.ic:hover{
  border-color:rgba(77,158,255,.45);
  box-shadow:0 0 20px rgba(77,158,255,.08);
}
.ic-h{
  padding:.45rem .85rem;
  font-family:var(--mono);font-size:.6rem;
  text-transform:uppercase;letter-spacing:.1em;color:var(--mu);
  background:var(--s3);border-bottom:1px solid var(--bd);
  display:flex;align-items:center;justify-content:space-between;
}
.badge{
  color:var(--ac);font-size:.58rem;
  background:rgba(77,158,255,.1);
  padding:.1rem .42rem;border-radius:4px;
  border:1px solid rgba(77,158,255,.2);
}
.ic-img{width:100%;display:block;}

/* ── Kotak analisis ── */
.analisis{
  background:var(--s2);border:1px solid var(--bd);
  border-left:3px solid var(--ac);
  border-radius:0 8px 8px 0;
  padding:1rem 1.25rem;margin:1.2rem 0;
  font-size:.82rem;color:var(--mu);line-height:1.8;
}
.analisis b{color:var(--tx);}
.analisis .hi{color:var(--ac);font-weight:600;}
.analisis .ok{color:var(--grn);font-weight:600;}

/* ── Tabel ── */
.tbl{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.79rem;margin-top:.6rem;}
.tbl th{
  padding:.4rem .8rem;border-bottom:2px solid var(--bd);
  color:var(--mu);text-align:left;font-weight:500;
  font-size:.62rem;text-transform:uppercase;letter-spacing:.08em;
}
.tbl td{padding:.38rem .8rem;border-bottom:1px solid var(--bd);color:var(--tx);}
.tbl tr:hover td{background:var(--s3);}
.tbl td.num{text-align:right;color:var(--ac);}
.tbl td.wkt{text-align:right;color:var(--mu);}
.tbl tr:last-child td{border-bottom:none;}

/* ── Notif ── */
.notif{
  background:rgba(251,146,60,.07);border:1px solid rgba(251,146,60,.3);
  border-radius:8px;padding:.65rem 1rem;color:var(--org);
  font-size:.8rem;margin:.5rem 0;font-family:var(--mono);
}
.notif-ok{
  background:rgba(74,222,128,.07);border:1px solid rgba(74,222,128,.3);
  border-radius:8px;padding:.65rem 1rem;color:var(--grn);
  font-size:.8rem;margin:.5rem 0;font-family:var(--mono);
}

/* ── Separator ── */
.sep{border:none;border-top:1px solid var(--bd);margin:1.3rem 0;}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:transparent!important;
  border-bottom:1px solid var(--bd);gap:.1rem;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important;color:var(--mu)!important;
  font-family:var(--mono);font-size:.76rem;
  padding:.4rem 1.1rem;border-radius:6px 6px 0 0;
  transition:color .15s;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--tx)!important;}
.stTabs [aria-selected="true"]{
  background:var(--s2)!important;color:var(--tx)!important;
  border:1px solid var(--bd)!important;
  border-bottom:1px solid var(--s2)!important;
}
.stTabs [data-baseweb="tab-panel"]{padding-top:1rem;}

/* ── Input ── */
[data-testid="stFileUploader"]{
  border:1px dashed var(--bd)!important;
  border-radius:10px!important;background:var(--s2)!important;
}
[data-testid="stFileUploader"]:hover{border-color:var(--ac)!important;}
.stRadio label span{color:var(--tx)!important;font-size:.85rem!important;}
.stCameraInput video{border-radius:8px!important;}
div.stButton>button{
  background:var(--ac);border:none;color:#000;
  font-weight:600;border-radius:8px;
}
/* Slider accent */
.stSlider [data-baseweb="slider"] [role="slider"]{background:var(--ac)!important;}

/* ── Empty state ── */
.kosong{text-align:center;padding:4rem 1rem;color:var(--mu);}
.kosong .ikon{font-size:3rem;margin-bottom:1rem;
  filter:drop-shadow(0 0 16px rgba(77,158,255,.3));}
.kosong h3{color:var(--tx);font-weight:600;font-size:1.1rem;margin:.5rem 0;}
.kosong p{font-family:var(--mono);font-size:.8rem;max-width:360px;margin:0 auto;}
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ev-wrap">
  <div class="ev-tag">Pengolahan Citra Digital &nbsp;/&nbsp; Studi Kasus</div>
  <div class="ev-title">Edge<em>Vision</em></div>
  <div class="ev-sub">Deteksi tepi wajah manusia menggunakan Operator Sobel &amp; Algoritma Canny</div>
  <div class="ev-pills">
    <span class="pill">Python 3 + OpenCV</span>
    <span class="pill">Haar Cascade Classifier</span>
    <span class="pill">Non-max Suppression</span>
    <span class="pill">Hysteresis Thresholding</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter")
    st.markdown("---")
    st.markdown("**Operator Sobel**")
    sobel_k = st.select_slider(
        "Ukuran kernel", options=[1, 3, 5, 7], value=3,
        help="Kernel makin besar → kontur lebih tebal, noise lebih halus"
    )
    st.markdown("---")
    st.markdown("**Algoritma Canny**")
    c_lo = st.slider("Threshold bawah", 0, 200,  50, step=5)
    c_hi = st.slider("Threshold atas",  0, 400, 150, step=10)
    st.markdown("---")
    st.markdown("""
**Panduan kernel Sobel:**  
`1` → gradien mentah  
`3` → detail halus *(default)*  
`5` → kontur lebih tebal  
`7` → hanya tepi dominan  

**Panduan threshold Canny:**  
Nilai rendah → lebih banyak tepi  
Nilai tinggi → hanya tepi kuat
""")
    st.markdown("---")
    st.markdown("""
**Metode deteksi otomatis:**  
Wajah: Haar Cascade frontal  
Mata: Region atas 55% wajah  
Mulut: Region bawah 40% wajah
""")

# ─── Input ───────────────────────────────────────────────────────────────────
mode = st.radio(
    "Sumber gambar",
    ["📁  Upload dari file / galeri", "📷  Kamera langsung"],
    horizontal=True, label_visibility="collapsed"
)

img_bgr = None
if "Upload" in mode:
    up = st.file_uploader(
        "Upload foto wajah (JPG/PNG/WEBP)",
        type=["jpg","jpeg","png","bmp","webp"],
        label_visibility="collapsed"
    )
    if up:
        img_bgr = load_image(up.read())
        if img_bgr is None:
            st.error("❌ File tidak dapat dibaca. Coba format JPG atau PNG.")
else:
    cam = st.camera_input("Ambil foto", label_visibility="collapsed")
    if cam:
        img_bgr = load_image(cam.read())

st.markdown('<hr class="sep"/>', unsafe_allow_html=True)

# ─── Empty state ─────────────────────────────────────────────────────────────
if img_bgr is None:
    st.markdown("""
    <div class="kosong">
      <div class="ikon">🔬</div>
      <h3>Siap untuk menganalisis</h3>
      <p>Upload foto wajah atau aktifkan kamera untuk memulai deteksi tepi secara real-time</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Proses ──────────────────────────────────────────────────────────────────
gray_arr            = to_gray(img_bgr)
gray_rgb            = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2RGB)
sobel_img, sob_ms   = apply_sobel(gray_arr, ksize=sobel_k)
canny_img, can_ms   = apply_canny(gray_arr, low=c_lo, high=c_hi)
sob_dens            = edge_density(sobel_img)
can_dens            = edge_density(canny_img)
h_px, w_px          = img_bgr.shape[:2]

# ─── Tabs ────────────────────────────────────────────────────────────────────
t1, t2 = st.tabs(["🔬  Deteksi Tepi", "👤  Deteksi Fitur Wajah"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETEKSI TEPI
# ════════════════════════════════════════════════════════════════════════════
with t1:

    # Metric cards
    st.markdown(
        '<div class="mc-row">'
        + met("Resolusi",            f"{w_px}×{h_px}", "px")
        + met("Waktu — Sobel",       f"{sob_ms:.1f}",  "ms")
        + met("Waktu — Canny",       f"{can_ms:.1f}",  "ms")
        + met("Kepadatan Tepi Sobel",f"{sob_dens:.1f}","%",  biru=True)
        + met("Kepadatan Tepi Canny",f"{can_dens:.1f}","%",  biru=True)
        + f'<div class="mc"><div class="mc-l">Rasio Sobel/Canny</div>'
          f'<div class="mc-v">'
          f'{(sob_dens/can_dens if can_dens>0 else 0):.1f}'
          f'<span class="mc-u">×</span></div></div>'
        + '</div>',
        unsafe_allow_html=True
    )

    # 4 gambar utama
    st.markdown(
        '<div class="ig ig-4">'
        + card(to_rgb(img_bgr), "Original",   "RGB")
        + card(gray_rgb,         "Grayscale",  "Konversi BGR→Gray")
        + card(sobel_img,         "Sobel",     f"Kernel {sobel_k}×{sobel_k}")
        + card(canny_img,         "Canny",     f"T [{c_lo}, {c_hi}]")
        + '</div>',
        unsafe_allow_html=True
    )

    # Analisis otomatis
    ratio    = sob_dens / can_dens if can_dens > 0 else 0
    lebih_cepat = "Canny" if can_ms < sob_ms else "Sobel"
    lebih_selektif = "Canny" if can_dens < sob_dens else "Sobel"
    st.markdown(f"""
    <div class="analisis">
    ▸ <b>Sobel</b> mendeteksi <b>{sob_dens:.1f}%</b> piksel sebagai tepi dalam <b>{sob_ms:.1f} ms</b>.
      Menghasilkan peta gradien kontinu (nilai abu-abu), sensitif terhadap perubahan intensitas di semua arah
      melalui kernel konvolusi sumbu-X dan sumbu-Y.<br>
    ▸ <b>Canny</b> mendeteksi <b>{can_dens:.1f}%</b> piksel sebagai tepi dalam <b>{can_ms:.1f} ms</b>.
      Menghasilkan tepi biner (hitam-putih) yang lebih presisi karena melalui 4 tahap:
      <em>Gaussian smoothing → Gradien Sobel → Non-maximum suppression → Hysteresis thresholding</em>.<br>
    ▸ Rasio kepadatan Sobel ÷ Canny = <span class="hi"><b>{ratio:.2f}×</b></span> —
      Sobel jauh lebih "berisik" karena tidak melakukan penipisan tepi.
      Canny mempertahankan hanya tepi yang signifikan secara perceptual.<br>
    ▸ Algoritma lebih cepat pada gambar ini: <span class="ok"><b>{lebih_cepat}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Tabel perbandingan parameter (data Bab 4)
    with st.expander("📊  Tabel perbandingan multi-parameter — data untuk Bab 4.3"):
        konfigurasi = [
            ("Sobel k=1",     *apply_sobel(gray_arr, ksize=1)),
            ("Sobel k=3",     *apply_sobel(gray_arr, ksize=3)),
            ("Sobel k=5",     *apply_sobel(gray_arr, ksize=5)),
            ("Sobel k=7",     *apply_sobel(gray_arr, ksize=7)),
            ("Canny T=30/90", *apply_canny(gray_arr, 30,  90)),
            ("Canny T=50/150",*apply_canny(gray_arr, 50, 150)),
            ("Canny T=80/200",*apply_canny(gray_arr, 80, 200)),
        ]
        baris = ""
        for nama, img_r, ms in konfigurasi:
            d = edge_density(img_r)
            baris += (f'<tr><td>{nama}</td>'
                      f'<td class="num">{d:.2f}%</td>'
                      f'<td class="wkt">{ms:.2f} ms</td></tr>')
        st.markdown(
            '<table class="tbl"><thead><tr>'
            '<th>Konfigurasi</th><th style="text-align:right">Kepadatan Tepi</th>'
            '<th style="text-align:right">Waktu Komputasi</th>'
            f'</tr></thead><tbody>{baris}</tbody></table>',
            unsafe_allow_html=True
        )
        st.caption("Data di atas dapat dijadikan tabel di Bab 4.3 makalah.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DETEKSI FITUR WAJAH
# ════════════════════════════════════════════════════════════════════════════
with t2:
    with st.spinner("Menjalankan deteksi wajah..."):
        det = detect_face(img_bgr)

    for n in det["notes"]:
        st.markdown(f'<div class="notif">⚠ {n}</div>', unsafe_allow_html=True)

    if not det["notes"] and det["face"] is not None:
        st.markdown('<div class="notif-ok">✓ Wajah terdeteksi — fitur berhasil diisolasi</div>',
                    unsafe_allow_html=True)

    # Baris 1: annotated + face crop
    b1 = [card(det["annotated"], "Anotasi Bounding Box", "Haar Cascade")]
    if det["face"] is not None:
        b1.append(card(det["face"], "Crop Wajah", "Region terbesar"))
    st.markdown(
        f'<div class="ig ig-2">{"".join(b1)}</div>',
        unsafe_allow_html=True
    )

    # Baris 2: mata + mulut
    b2 = []
    if det.get("right_eye") is not None:
        b2.append(card(det["right_eye"], "Mata Kanan", "crop"))
    if det.get("left_eye") is not None:
        b2.append(card(det["left_eye"],  "Mata Kiri",  "crop"))
    if det.get("mouth") is not None:
        b2.append(card(det["mouth"],     "Area Mulut", "crop"))
    if b2:
        kls = ["","ig-1","ig-2","ig-3","ig-3"][min(len(b2),4)]
        st.markdown(
            f'<div class="ig {kls}">{"".join(b2)}</div>',
            unsafe_allow_html=True
        )

    # Deteksi tepi pada crop fitur
    if det["face"] is not None:
        with st.expander("🔬  Deteksi tepi pada setiap fitur wajah"):
            target_list = [
                ("Wajah",      det["face"]),
                ("Mata Kanan", det.get("right_eye")),
                ("Mata Kiri",  det.get("left_eye")),
                ("Area Mulut", det.get("mouth")),
            ]
            for nama, crop_rgb in target_list:
                if crop_rgb is None:
                    continue
                bgr_c  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
                gr_c   = to_gray(bgr_c)
                si, _  = apply_sobel(gr_c, ksize=sobel_k)
                ci, _  = apply_canny(gr_c, low=c_lo, high=c_hi)
                sd, cd = edge_density(si), edge_density(ci)
                st.markdown(f"""
                <div style="font-family:var(--mono);font-size:.72rem;
                  color:var(--mu);margin:.8rem 0 .4rem;">
                  {nama} &nbsp;—&nbsp;
                  Sobel: <b style="color:#E8EFF8">{sd:.1f}%</b> &nbsp;|&nbsp;
                  Canny: <b style="color:#4D9EFF">{cd:.1f}%</b>
                </div>""", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="ig ig-3">'
                    + card(crop_rgb, f"{nama} — Asli",  "RGB")
                    + card(si,       f"{nama} — Sobel", f"k={sobel_k}")
                    + card(ci,       f"{nama} — Canny", f"T[{c_lo},{c_hi}]")
                    + '</div>',
                    unsafe_allow_html=True
                )

    # Catatan teknis
    st.markdown("""
    <div class="analisis" style="margin-top:1.2rem;">
    ▸ <b>Deteksi wajah</b>: Haar Cascade frontal — scaleFactor 1.1, minNeighbors 5, minSize 80×80<br>
    ▸ <b>Deteksi mata</b>: Haar Eye Cascade — dibatasi pada 55% area atas wajah
      untuk menekan <em>false positive</em> dari hidung/alis<br>
    ▸ <b>Deteksi mulut</b>: Haar Smile Cascade — dibatasi pada 40% area bawah wajah;
      jika senyuman tidak terdeteksi, digunakan estimasi geometri proporsional sebagai <em>fallback</em>
    </div>
    """, unsafe_allow_html=True)
