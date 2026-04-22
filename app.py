import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import io
import time

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Konversi RGB → Grayscale",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Background */
.stApp {
    background: #0B0B14;
    color: #D4D4E8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #08080F !important;
    border-right: 1px solid #1C1C30 !important;
}
[data-testid="stSidebar"] * {
    color: #A0A0C0 !important;
}

/* Titles */
h1, h2, h3 { font-family: 'Sora', sans-serif !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #10101E;
    border: 1px solid #1E1E35;
    border-radius: 8px;
    padding: 12px;
}

/* Code blocks */
code {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0A0A18 !important;
    color: #79D4B5 !important;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Expander */
[data-testid="stExpander"] {
    background: #0E0E1C;
    border: 1px solid #1A1A30;
    border-radius: 8px;
}

/* Radio buttons */
[data-testid="stRadio"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
}

/* Divider styling */
hr { border-color: #1A1A30 !important; }

/* Info / warning / success boxes */
.stAlert {
    background: #0D0D1E !important;
    border-radius: 8px !important;
}

/* Header badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    letter-spacing: 1px;
}

/* Formula box */
.formula-box {
    background: #060612;
    border: 1px solid #1E1E38;
    border-left: 3px solid #7C6AFF;
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #B0A8FF;
    margin: 8px 0;
}

/* Pixel table */
.pixel-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    background: #0A0A18;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 4px 0;
    border: 1px solid #1A1A2C;
}

/* Section header */
.section-header {
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #444466;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
}

.stButton > button {
    background: #7C6AFF22;
    border: 1px solid #7C6AFF66;
    color: #B0A8FF;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #7C6AFF44;
    border-color: #7C6AFF;
    color: white;
}

.stDownloadButton > button {
    width: 100%;
    background: #1E2A1A;
    border: 1px solid #3A6A2A;
    color: #7ACF5A;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# ALGORITHM DEFINITIONS
# ─────────────────────────────────────────────

METHODS = {
    "1. Rata-rata (Average)": {
        "id": "average",
        "formula": "Gray = (R + G + B) / 3",
        "formula_latex": "G = \\frac{R + G + B}{3}",
        "desc": "Metode paling sederhana. Setiap kanal RGB diberi bobot yang sama (1/3). Mudah dipahami namun kurang akurat secara persepsi karena mata manusia tidak sama sensitifnya terhadap ketiga warna.",
        "ref": "Gonzalez & Woods, Digital Image Processing, Ch.6",
        "color": "#FF6B6B",
        "func": lambda r, g, b: (r.astype(np.float64) + g.astype(np.float64) + b.astype(np.float64)) / 3.0,
        "calc_str": lambda r, g, b: f"({r} + {g} + {b}) / 3 = {(r+g+b)/3:.4f} ≈ {round((r+g+b)/3)}",
    },
    "2. Luminosity / Weighted (BT.601)": {
        "id": "bt601",
        "formula": "Gray = 0.299·R + 0.587·G + 0.114·B",
        "formula_latex": "G = 0.299R + 0.587G + 0.114B",
        "desc": "Standar ITU-R BT.601 (NTSC/PAL). Bobot berbeda karena sensitivitas mata: hijau paling sensitif, merah sedang, biru paling kurang. Ini metode paling umum diajarkan di mata kuliah PAID.",
        "ref": "ITU-R BT.601 | Gonzalez & Woods, Digital Image Processing",
        "color": "#4ECDC4",
        "func": lambda r, g, b: 0.299 * r.astype(np.float64) + 0.587 * g.astype(np.float64) + 0.114 * b.astype(np.float64),
        "calc_str": lambda r, g, b: f"0.299×{r} + 0.587×{g} + 0.114×{b} = {0.299*r + 0.587*g + 0.114*b:.4f} ≈ {round(0.299*r + 0.587*g + 0.114*b)}",
    },
    "3. Luminance (BT.709 / HDTV)": {
        "id": "bt709",
        "formula": "Gray = 0.2126·R + 0.7152·G + 0.0722·B",
        "formula_latex": "G = 0.2126R + 0.7152G + 0.0722B",
        "desc": "Standar ITU-R BT.709 untuk HDTV dan sRGB. Lebih akurat untuk layar modern. Bobot hijau lebih dominan dibanding BT.601. Digunakan pada sistem pengolahan citra digital modern.",
        "ref": "ITU-R BT.709 | IEC 61966-2-1 (sRGB)",
        "color": "#45B7D1",
        "func": lambda r, g, b: 0.2126 * r.astype(np.float64) + 0.7152 * g.astype(np.float64) + 0.0722 * b.astype(np.float64),
        "calc_str": lambda r, g, b: f"0.2126×{r} + 0.7152×{g} + 0.0722×{b} = {0.2126*r + 0.7152*g + 0.0722*b:.4f} ≈ {round(0.2126*r + 0.7152*g + 0.0722*b)}",
    },
    "4. Nilai Tengah / Lightness": {
        "id": "lightness",
        "formula": "Gray = (max(R,G,B) + min(R,G,B)) / 2",
        "formula_latex": "G = \\frac{\\max(R,G,B) + \\min(R,G,B)}{2}",
        "desc": "Mengambil rata-rata antara nilai piksel maksimum dan minimum. Merupakan komponen Lightness dalam model warna HSL. Mengabaikan kanal tengah sehingga kehilangan sebagian informasi.",
        "ref": "Foley et al., Computer Graphics: Principles and Practice",
        "color": "#F7DC6F",
        "func": lambda r, g, b: (np.maximum(np.maximum(r, g), b).astype(np.float64) + np.minimum(np.minimum(r, g), b).astype(np.float64)) / 2.0,
        "calc_str": lambda r, g, b: f"(max({r},{g},{b}) + min({r},{g},{b})) / 2 = ({max(r,g,b)} + {min(r,g,b)}) / 2 = {(max(r,g,b)+min(r,g,b))/2:.4f} ≈ {round((max(r,g,b)+min(r,g,b))/2)}",
    },
    "5. Nilai Minimum (Min)": {
        "id": "min",
        "formula": "Gray = min(R, G, B)",
        "formula_latex": "G = \\min(R, G, B)",
        "desc": "Hanya menggunakan nilai kanal terkecil. Hasil cenderung gelap karena selalu mengambil nilai terendah. Berguna untuk analisis bayangan atau deteksi area gelap pada citra.",
        "ref": "Pratt, Digital Image Processing, 4th ed.",
        "color": "#A29BFE",
        "func": lambda r, g, b: np.minimum(np.minimum(r, g), b).astype(np.float64),
        "calc_str": lambda r, g, b: f"min({r}, {g}, {b}) = {min(r,g,b)}",
    },
    "6. Nilai Maksimum (Max)": {
        "id": "max",
        "formula": "Gray = max(R, G, B)",
        "formula_latex": "G = \\max(R, G, B)",
        "desc": "Hanya menggunakan nilai kanal terbesar. Hasil cenderung terang. Digunakan pada kasus tertentu seperti deteksi highlight atau area terang pada citra.",
        "ref": "Pratt, Digital Image Processing, 4th ed.",
        "color": "#FFEAA7",
        "func": lambda r, g, b: np.maximum(np.maximum(r, g), b).astype(np.float64),
        "calc_str": lambda r, g, b: f"max({r}, {g}, {b}) = {max(r,g,b)}",
    },
    "7. Kanal Merah Saja (R)": {
        "id": "red",
        "formula": "Gray = R",
        "formula_latex": "G = R",
        "desc": "Hanya menggunakan komponen merah. Berguna saat ingin menganalisis distribusi warna merah pada citra. Contoh aplikasi: citra medis (pembuluh darah), deteksi objek merah.",
        "ref": "Gonzalez & Woods, Digital Image Processing, Ch.6",
        "color": "#FF4757",
        "func": lambda r, g, b: r.astype(np.float64),
        "calc_str": lambda r, g, b: f"Gray = R = {r}",
    },
    "8. Kanal Hijau Saja (G)": {
        "id": "green",
        "formula": "Gray = G",
        "formula_latex": "G = G",
        "desc": "Hanya menggunakan komponen hijau. Kanal hijau mengandung paling banyak informasi luminance sehingga hasilnya paling mirip dengan persepsi visual manusia dibanding kanal R dan B.",
        "ref": "Gonzalez & Woods, Digital Image Processing, Ch.6",
        "color": "#2ED573",
        "func": lambda r, g, b: g.astype(np.float64),
        "calc_str": lambda r, g, b: f"Gray = G = {g}",
    },
    "9. Kanal Biru Saja (B)": {
        "id": "blue",
        "formula": "Gray = B",
        "formula_latex": "G = B",
        "desc": "Hanya menggunakan komponen biru. Mata manusia paling tidak sensitif terhadap biru sehingga hasilnya cenderung kurang kontras. Berguna untuk analisis komponen biru (misal: langit, air).",
        "ref": "Gonzalez & Woods, Digital Image Processing, Ch.6",
        "color": "#1E90FF",
        "func": lambda r, g, b: b.astype(np.float64),
        "calc_str": lambda r, g, b: f"Gray = B = {b}",
    },
    "10. Dekomposisi Perceptual (Luma)": {
        "id": "luma",
        "formula": "Y' = 0.299·R + 0.587·G + 0.114·B  (dengan gamma encoding)",
        "formula_latex": "Y' = 0.299R' + 0.587G' + 0.114B'",
        "desc": "Luma (Y') adalah versi BT.601 yang sudah mempertimbangkan gamma encoding layar. Berbeda dengan luminance linear, Luma bekerja langsung pada nilai piksel terkoreksi-gamma. Inilah yang digunakan JPEG, MPEG, dll.",
        "ref": "Rec. ITU-R BT.601-7 | JPEG Standard (ISO 10918)",
        "color": "#FFA502",
        "func": lambda r, g, b: 0.299 * r.astype(np.float64) + 0.587 * g.astype(np.float64) + 0.114 * b.astype(np.float64),
        "calc_str": lambda r, g, b: f"Y' = 0.299×{r} + 0.587×{g} + 0.114×{b} = {0.299*r + 0.587*g + 0.114*b:.4f} ≈ {round(0.299*r + 0.587*g + 0.114*b)}",
    },
}


# ─────────────────────────────────────────────
# CONVERSION FUNCTION (MANUAL — NO CV2/SKIMAGE)
# ─────────────────────────────────────────────

def convert_to_gray_manual(img_array: np.ndarray, method_key: str) -> np.ndarray:
    """
    Konversi manual RGB → Grayscale.
    Semua operasi menggunakan numpy; TIDAK menggunakan cv2.cvtColor atau skimage.
    """
    r = img_array[:, :, 0].astype(np.float64)
    g = img_array[:, :, 1].astype(np.float64)
    b = img_array[:, :, 2].astype(np.float64)

    gray_float = METHODS[method_key]["func"](r, g, b)

    # Clamp ke rentang [0, 255] lalu konversi ke uint8
    gray_clipped = np.clip(gray_float, 0, 255)
    gray_uint8 = np.round(gray_clipped).astype(np.uint8)
    return gray_uint8


def get_pixel_calc(img_array: np.ndarray, x: int, y: int, method_key: str) -> str:
    """Tampilkan perhitungan satu piksel secara detail."""
    r = int(img_array[y, x, 0])
    g = int(img_array[y, x, 1])
    b = int(img_array[y, x, 2])
    return METHODS[method_key]["calc_str"](r, g, b)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px; text-align:center;'>
        <div style='font-size:32px;'>🎓</div>
        <div style='font-family: Sora, sans-serif; font-size:16px; font-weight:800; color:#fff; margin-top:8px;'>
            Konversi RGB → Grayscale
        </div>
        <div style='font-size:10px; color:#444; letter-spacing:2px; margin-top:4px;font-family:monospace;'>
            PENGOLAHAN CITRA DIGITAL
        </div>
    </div>
    <hr style='margin: 12px 0; border-color:#1C1C30;'>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">📁 Upload Gambar</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"])

    st.markdown('<hr style="margin:16px 0; border-color:#1C1C30;">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">⚙️ Pilih Metode</p>', unsafe_allow_html=True)
    selected_method = st.radio("", list(METHODS.keys()), label_visibility="collapsed")

    st.markdown('<hr style="margin:16px 0; border-color:#1C1C30;">', unsafe_allow_html=True)
    st.markdown('<p class="section-header">🔬 Mode Analisis</p>', unsafe_allow_html=True)
    show_compare = st.checkbox("Bandingkan semua metode", value=False)
    show_histogram = st.checkbox("Tampilkan histogram", value=True)
    show_pixel_table = st.checkbox("Tabel piksel sampel", value=True)

    if uploaded:
        st.markdown('<hr style="margin:16px 0; border-color:#1C1C30;">', unsafe_allow_html=True)
        st.markdown('<p class="section-header">ℹ️ Info Citra</p>', unsafe_allow_html=True)
        img_tmp = Image.open(uploaded).convert("RGB")
        arr_tmp = np.array(img_tmp)
        st.markdown(f"""
        <div style='font-family:monospace; font-size:12px; color:#666; line-height:2;'>
        Dimensi &nbsp;: {arr_tmp.shape[1]} × {arr_tmp.shape[0]} px<br>
        Kanal &nbsp;&nbsp;&nbsp;: {arr_tmp.shape[2]} (RGB)<br>
        Total px : {arr_tmp.shape[0]*arr_tmp.shape[1]:,}<br>
        Tipe data: uint8 (0–255)
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────

# Header
st.markdown("""
<div style='padding: 24px 0 8px;'>
    <h1 style='font-family: Sora, sans-serif; font-size: 28px; font-weight: 800; color: #fff; margin:0;'>
        Konversi Citra RGB ke <span style='color:#7C6AFF;'>Gray Level</span>
    </h1>
    <p style='color:#555; font-size:13px; margin-top:6px; font-family:monospace; letter-spacing:1px;'>
        Mata Kuliah: Pengolahan & Analisis Citra Digital &nbsp;·&nbsp; Perhitungan Manual Piksel-per-Piksel
    </p>
</div>
""", unsafe_allow_html=True)

if not uploaded:
    # Landing info
    st.markdown("""
    <div style='background:#0E0E1C; border:1px solid #1A1A30; border-radius:12px; padding:40px; text-align:center; margin-top:20px;'>
        <div style='font-size:48px; margin-bottom:16px;'>🖼️</div>
        <div style='font-family: Sora, sans-serif; font-size:18px; color:#888; margin-bottom:8px;'>
            Upload gambar menggunakan panel kiri untuk memulai
        </div>
        <div style='font-size:12px; color:#444; font-family:monospace;'>
            Format: PNG · JPG · BMP · WEBP · TIFF
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Method overview table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📚 Semua Metode yang Tersedia</p>', unsafe_allow_html=True)
    rows = []
    for name, m in METHODS.items():
        rows.append({"Metode": name, "Formula": m["formula"], "Referensi": m["ref"]})
    df_methods = pd.DataFrame(rows)
    st.dataframe(df_methods, use_container_width=True, hide_index=True,
                 column_config={
                     "Metode": st.column_config.TextColumn(width=280),
                     "Formula": st.column_config.TextColumn(width=320),
                     "Referensi": st.column_config.TextColumn(width=320),
                 })
    st.stop()


# ─────────────────────────────────────────────
# PROCESS IMAGE
# ─────────────────────────────────────────────

img_pil = Image.open(uploaded).convert("RGB")
img_arr = np.array(img_pil)
H, W = img_arr.shape[:2]

# Resize if too large (performance)
MAX_DIM = 800
if max(H, W) > MAX_DIM:
    ratio = MAX_DIM / max(H, W)
    new_w, new_h = int(W * ratio), int(H * ratio)
    img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
    img_arr = np.array(img_pil)
    H, W = img_arr.shape[:2]
    st.info(f"ℹ️ Gambar diresize ke {W}×{H} px untuk performa optimal.")

method = METHODS[selected_method]

# Run conversion
with st.spinner(f"Menghitung konversi menggunakan **{selected_method}**..."):
    gray_arr = convert_to_gray_manual(img_arr, selected_method)
    gray_pil = Image.fromarray(gray_arr, mode="L")
    time.sleep(0.1)  # brief pause for UX

# ─────────────────────────────────────────────
# METODE INFO CARD
# ─────────────────────────────────────────────

col_info1, col_info2 = st.columns([3, 2])
with col_info1:
    st.markdown(f"""
    <div style='background:#0E0E1C; border:1px solid {method["color"]}33;
                border-left: 3px solid {method["color"]}; border-radius:8px; padding:16px 20px; margin-top:16px;'>
        <div style='font-size:16px; font-weight:700; color:{method["color"]}; margin-bottom:6px;'>
            {selected_method}
        </div>
        <div style='font-size:12px; color:#888; line-height:1.7; margin-bottom:10px;'>
            {method["desc"]}
        </div>
        <div class="formula-box">{method["formula"]}</div>
        <div style='font-size:10px; color:#444; font-family:monospace; margin-top:8px;'>
            📖 Referensi: {method["ref"]}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_info2:
    st.markdown("<div style='margin-top:16px;'>", unsafe_allow_html=True)
    st.metric("Lebar (Width)", f"{W} px")
    st.metric("Tinggi (Height)", f"{H} px")
    st.metric("Total Piksel", f"{W*H:,}")
    st.metric("Rata-rata Gray", f"{int(gray_arr.mean())}")

# ─────────────────────────────────────────────
# SIDE BY SIDE IMAGE COMPARISON
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">🔲 Perbandingan Citra</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"<div style='font-size:11px; letter-spacing:2px; color:#555; font-family:monospace; margin-bottom:8px;'>CITRA ASLI (RGB)</div>", unsafe_allow_html=True)
    st.image(img_pil, use_container_width=True)
with col2:
    st.markdown(f"<div style='font-size:11px; letter-spacing:2px; color:{method['color']}; font-family:monospace; margin-bottom:8px;'>HASIL GRAYSCALE — {method['id'].upper()}</div>", unsafe_allow_html=True)
    st.image(gray_pil, use_container_width=True)

# Download button
buf = io.BytesIO()
gray_pil.save(buf, format="PNG")
st.download_button(
    label="⬇️  Download Hasil Grayscale (.png)",
    data=buf.getvalue(),
    file_name=f"grayscale_{method['id']}.png",
    mime="image/png",
    use_container_width=True,
)


# ─────────────────────────────────────────────
# HISTOGRAM
# ─────────────────────────────────────────────

if show_histogram:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Histogram Distribusi Intensitas</p>', unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.patch.set_facecolor("#0A0A14")

    titles = ["Kanal Merah (R)", "Kanal Hijau (G)", "Kanal Biru (B)"]
    colors_hist = ["#FF4757", "#2ED573", "#1E90FF"]
    channels = [img_arr[:, :, i].flatten() for i in range(3)]

    for ax, data, title, c in zip(axes[:3], channels, titles, colors_hist):
        ax.set_facecolor("#0D0D1C")
        ax.hist(data, bins=64, color=c, alpha=0.8, edgecolor="none")
        ax.set_title(title, color="#AAA", fontsize=10, fontfamily="monospace")
        ax.tick_params(colors="#444", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1A1A30")
        ax.set_xlim(0, 255)

    plt.tight_layout(pad=1.5)
    st.pyplot(fig)
    plt.close()

    # Gray histogram
    fig2, ax2 = plt.subplots(figsize=(14, 2.5))
    fig2.patch.set_facecolor("#0A0A14")
    ax2.set_facecolor("#0D0D1C")
    ax2.hist(gray_arr.flatten(), bins=128, color=method["color"], alpha=0.9, edgecolor="none")
    ax2.set_title(f"Distribusi Gray Level — {selected_method}", color="#AAA", fontsize=11, fontfamily="monospace")
    ax2.tick_params(colors="#444", labelsize=8)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1A1A30")
    ax2.set_xlim(0, 255)
    ax2.set_ylabel("Jumlah Piksel", color="#666", fontsize=9)
    ax2.set_xlabel("Nilai Intensitas (0–255)", color="#666", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()


# ─────────────────────────────────────────────
# PIXEL SAMPLE TABLE — MANUAL CALCULATION
# ─────────────────────────────────────────────

if show_pixel_table:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">🧮 Tabel Perhitungan Manual Piksel Sampel</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:12px; color:#666; margin-bottom:12px;'>
    Berikut adalah perhitungan manual untuk 10 piksel sampel menggunakan metode <b style='color:{method["color"]};'>{selected_method}</b>.
    Setiap baris menunjukkan langkah demi langkah konversi nilai RGB → Gray.
    </div>
    """, unsafe_allow_html=True)

    # Sample coordinates (spread across image)
    np.random.seed(42)
    sample_coords = [
        (np.random.randint(0, W), np.random.randint(0, H)) for _ in range(10)
    ]
    # Also include corners and center
    sample_coords = [(0,0),(W//4,H//4),(W//2,H//2),(3*W//4,3*H//4),(W-1,H-1),
                     (W//2,0),(0,H//2),(W-1,H//2),(W//3,H//3),(2*W//3,2*H//3)]

    rows = []
    for (x, y) in sample_coords:
        x = min(x, W-1)
        y = min(y, H-1)
        r = int(img_arr[y, x, 0])
        g = int(img_arr[y, x, 1])
        b = int(img_arr[y, x, 2])
        gray_val = int(gray_arr[y, x])
        calc = method["calc_str"](r, g, b)
        rows.append({
            "Koordinat (x,y)": f"({x}, {y})",
            "R": r,
            "G": g,
            "B": b,
            "Perhitungan": calc,
            "Hasil Gray": gray_val,
        })

    df_pixels = pd.DataFrame(rows)
    st.dataframe(df_pixels, use_container_width=True, hide_index=True,
                 column_config={
                     "Koordinat (x,y)": st.column_config.TextColumn(width=120),
                     "R": st.column_config.NumberColumn(width=60),
                     "G": st.column_config.NumberColumn(width=60),
                     "B": st.column_config.NumberColumn(width=60),
                     "Perhitungan": st.column_config.TextColumn(width=400),
                     "Hasil Gray": st.column_config.NumberColumn(width=100),
                 })

    # Download tabel
    csv_buf = df_pixels.to_csv(index=False).encode()
    st.download_button("⬇️  Download Tabel Perhitungan (.csv)", csv_buf,
                       file_name=f"perhitungan_{method['id']}.csv", mime="text/csv")


# ─────────────────────────────────────────────
# COMPARE ALL METHODS
# ─────────────────────────────────────────────

if show_compare:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">⚖️ Perbandingan Semua Metode</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:12px; color:#555; margin-bottom:16px;'>
    Setiap metode menghasilkan distribusi gray level yang berbeda. Perhatikan perbedaan kontras dan detail pada tiap hasil.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Memproses semua metode..."):
        all_grays = {}
        for mname in METHODS:
            all_grays[mname] = convert_to_gray_manual(img_arr, mname)

    # Display 5 per row
    method_names = list(METHODS.keys())
    for row_start in range(0, len(method_names), 5):
        row_methods = method_names[row_start:row_start+5]
        cols = st.columns(len(row_methods))
        for col, mname in zip(cols, row_methods):
            with col:
                m = METHODS[mname]
                gray_img = Image.fromarray(all_grays[mname], mode="L")
                st.image(gray_img, use_container_width=True)
                st.markdown(f"""
                <div style='text-align:center; margin-top:4px;'>
                    <div style='font-size:9px; color:{m["color"]}; font-family:monospace; line-height:1.4;'>
                        {mname.split(". ")[1] if ". " in mname else mname}
                    </div>
                    <div style='font-size:9px; color:#444; font-family:monospace;'>
                        avg={int(all_grays[mname].mean())} | std={int(all_grays[mname].std())}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Stats comparison table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📈 Statistik Perbandingan</p>', unsafe_allow_html=True)
    stat_rows = []
    for mname, garr in all_grays.items():
        stat_rows.append({
            "Metode": mname,
            "Min": int(garr.min()),
            "Max": int(garr.max()),
            "Rata-rata": f"{garr.mean():.2f}",
            "Std Dev": f"{garr.std():.2f}",
            "Median": f"{np.median(garr):.1f}",
        })
    df_stats = pd.DataFrame(stat_rows)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# PENJELASAN TEORI (EXPANDER)
# ─────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📖 Teori: Mengapa Konversi RGB → Grayscale Perlu Bobot Berbeda?"):
    st.markdown("""
    ### Sensitivitas Mata Manusia terhadap Warna

    Mata manusia memiliki **tiga jenis sel kerucut (cone cells)** yang sensitif terhadap panjang gelombang berbeda:

    | Warna | Bobot (BT.601) | Penjelasan |
    |-------|---------------|------------|
    | 🔴 Merah | **0.299** | Sensitifitas sedang, ~30% kontribusi |
    | 🟢 Hijau | **0.587** | Paling sensitif, ~59% kontribusi — mendominasi persepsi brightness |
    | 🔵 Biru  | **0.114** | Paling tidak sensitif, ~11% kontribusi |

    ### Model Persamaan Konversi

    Secara umum, persamaan konversi dapat ditulis:

    ```
    Gray = w_R × R + w_G × G + w_B × B
    ```

    di mana `w_R + w_G + w_B = 1` (untuk metode berbasis bobot).

    ### Perbedaan Standar BT.601 vs BT.709

    - **BT.601** didesain untuk TV analog (NTSC/PAL), fosfor CRT lama
    - **BT.709** didesain untuk HDTV dan layar LCD modern dengan gamut warna yang lebih luas
    - Keduanya valid; pilih BT.709 untuk citra digital modern

    ### Proses Manual Piksel-per-Piksel

    Pada aplikasi ini, konversi dilakukan **tanpa library image processing** (tidak menggunakan `cv2.cvtColor` atau fungsi serupa).
    Semua perhitungan dilakukan langsung pada array numpy piksel demi piksel:

    ```python
    # Contoh BT.601
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    ```
    """)

# Footer
st.markdown("""
<hr style='margin-top:40px; border-color:#1A1A30;'>
<div style='text-align:center; padding:16px 0; font-family:monospace; font-size:11px; color:#333; letter-spacing:1px;'>
    PENGOLAHAN & ANALISIS CITRA DIGITAL · Konversi Manual RGB → Grayscale · Built with Python + Streamlit
</div>
""", unsafe_allow_html=True)
