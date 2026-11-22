"""Aplikasi web (Streamlit): Simulasi System Dynamics PM10 ISPU DKI1 Jakarta."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import sd_model as m

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulasi SD Kualitas Udara Jakarta",
    page_icon="🌫️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(160deg, #060c18 0%, #0b1829 55%, #071220 100%);
    color: #ddeeff;
}

/* ── Header ── */
.main-header {
    font-size: 2.2rem; font-weight: 700; line-height: 1.1;
    background: linear-gradient(90deg, #48cae4, #00b4d8, #0096c7, #0077b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 2px;
}
.sub-caption {
    color: #6ec6e0; font-size: 0.82rem; margin: 0 0 1.4rem 0;
    letter-spacing: 0.03em;
}

/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 14px; margin-bottom: 1.2rem; }
.kpi-card {
    flex: 1; background: linear-gradient(135deg, rgba(0,180,216,.12), rgba(0,119,182,.07));
    border: 1px solid rgba(0,180,216,.28); border-radius: 14px;
    padding: 1rem 1.1rem; text-align: center; backdrop-filter: blur(6px);
    transition: border-color .2s;
}
.kpi-card:hover { border-color: rgba(0,180,216,.55); }
.kpi-label { font-size: .7rem; color: #6ec6e0; text-transform: uppercase;
             letter-spacing: .08em; margin-bottom: 4px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #48cae4; line-height: 1.1; }
.kpi-unit  { font-size: .72rem; color: #6ec6e0; margin-top: 2px; }

/* ── Status badge ── */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: .72rem; font-weight: 600; letter-spacing: .05em;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09162a 0%, #060f1e 100%) !important;
    border-right: 1px solid rgba(0,180,216,.18);
}
[data-testid="stSidebar"] label   { color: #7ecfe3 !important; font-size: .82rem !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3      { color: #48cae4 !important; }
[data-testid="stSidebar"] .stSlider > div [data-testid="stThumbValue"] {
    background: #0096c7 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,180,216,.06); border-radius: 10px; gap: 4px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; color: #6ec6e0; font-weight: 500; padding: 8px 22px;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg,#0077b6,#00b4d8) !important; color: #fff !important;
}

/* ── Divider ── */
.hdiv {
    border: none; height: 1px; margin: 1.2rem 0;
    background: linear-gradient(90deg, transparent, rgba(0,180,216,.4), transparent);
}

/* ── Info box ── */
.info-box {
    background: rgba(0,180,216,.08); border-left: 3px solid #00b4d8;
    border-radius: 0 8px 8px 0; padding: .65rem 1rem;
    margin: .8rem 0; font-size: .83rem; color: #b0d8ed;
}

/* ── Metric table ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,180,216,.2) !important; border-radius: 10px !important;
}

/* ── Footer ── */
.footer {
    text-align: center; color: rgba(110,198,224,.35);
    font-size: .72rem; margin-top: 3rem; padding-top: 1rem;
    border-top: 1px solid rgba(0,180,216,.1);
}
</style>
""", unsafe_allow_html=True)

# ── ISPU Categories ───────────────────────────────────────────────────────────
ISPU_CAT = [
    (0,   50,  "Baik",               "#00c48a"),
    (51,  100, "Sedang",             "#ffd166"),
    (101, 199, "Tidak Sehat",        "#ff7b30"),
    (200, 999, "Sangat Tidak Sehat", "#ef233c"),
]

def ispu_cat(val):
    for lo, hi, label, col in ISPU_CAT:
        if lo <= val <= hi:
            return label, col
    return "N/A", "#888"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load():
    d      = m.load_data("ispu_dki1.csv")
    obs    = m.monthly_series(d, "pm10")
    params = json.load(open("params.json"))
    coef   = m.fit_share_baik(d)
    return d, obs, params, coef

d, obs, params, coef = load()
base = [params[k] for k in m.PARAM_NAMES]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Parameter Model")
    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
    E0   = st.slider("E₀ · Emisi dasar (indeks/hari)",         1.0,  15.0, float(round(base[0], 2)), 0.1)
    g    = st.slider("g · Pertumbuhan emisi / tahun",          -0.03, 0.05, float(round(base[1], 3)), 0.001, format="%.3f")
    tau0 = st.slider("τ₀ · Waktu tinggal polutan (hari)",       2.0,  20.0, float(round(base[2], 2)), 0.1)
    s    = st.slider("s · Amplitudo musiman dispersi",           0.0,   0.6, float(round(base[3], 3)), 0.01)
    phi  = st.slider("φ · Puncak musim (hari ke-)",              0,   365, int(round(base[4])), 1)
    kmax = st.slider("Kmax · Umpan balik kebijakan",             0.0,   0.6, float(round(base[5], 3)), 0.01)

    st.markdown("## 🏛️ Skenario Kebijakan")
    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
    cut      = st.slider("Pengurangan emisi (%)", 0, 50, 20, 5) / 100
    cut_from = st.date_input("Mulai berlaku", pd.Timestamp("2025-03-01"))
    end_year = st.slider("Simulasi sampai tahun", 2026, 2035, 2030)

    st.markdown("## 🎨 Skala ISPU")
    st.markdown('<hr class="hdiv">', unsafe_allow_html=True)
    for lo, hi, label, col in ISPU_CAT:
        hi_str = str(hi) if hi < 999 else "∞"
        st.markdown(
            f'<span class="badge" style="background:rgba(0,0,0,.25);'
            f'color:{col};border:1px solid {col}">'
            f'{label} &nbsp;{lo}–{hi_str}</span>',
            unsafe_allow_html=True,
        )

p = [E0, g, tau0, s, phi, kmax]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🌫️ Simulasi Kualitas Udara Jakarta</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-caption">System Dynamics · PM10 ISPU · Stasiun DKI1 Bunderan HI · '
    'Data 2010–2025 · Mata kuliah Pemodelan &amp; Simulasi</p>',
    unsafe_allow_html=True,
)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
pm10_30d = d["pm10"].dropna().iloc[-30:].mean()
cat_30d, col_30d = ispu_cat(pm10_30d)
pct_baik  = (d["pm10"].dropna() <= 50).mean() * 100
obs_ok    = obs.dropna()
trend_val = obs_ok.iloc[-12:].mean() - obs_ok.iloc[-24:-12].mean()
trend_str = f"+{trend_val:.1f}" if trend_val > 0 else f"{trend_val:.1f}"
trend_col = "#ff7b30" if trend_val > 0 else "#00c48a"

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">PM10 rata-rata 30 hari terakhir</div>
    <div class="kpi-value">{pm10_30d:.1f}</div>
    <div class="kpi-unit">indeks ISPU &nbsp;·&nbsp;
      <span class="badge" style="color:{col_30d};border:1px solid {col_30d};
        background:rgba(0,0,0,.25)">{cat_30d}</span>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Hari kategori Baik (historis)</div>
    <div class="kpi-value">{pct_baik:.1f}<span style="font-size:1.1rem">%</span></div>
    <div class="kpi-unit">PM10 ≤ 50</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Tren PM10 (12 bulan terakhir)</div>
    <div class="kpi-value" style="color:{trend_col}">{trend_str}</div>
    <div class="kpi-unit">vs 12 bulan sebelumnya</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Total data bulanan tersedia</div>
    <div class="kpi-value">{obs_ok.count()}</div>
    <div class="kpi-unit">bulan · 2010–2025</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="hdiv">', unsafe_allow_html=True)

# ── Chart helper ──────────────────────────────────────────────────────────────
def dark_fig(w=11, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#07111f")
    ax.set_facecolor("#0b1829")
    for sp in ax.spines.values():
        sp.set_color("#17395e")
    ax.tick_params(colors="#6ec6e0", labelsize=9)
    ax.yaxis.label.set_color("#6ec6e0")
    ax.xaxis.label.set_color("#6ec6e0")
    return fig, ax

def add_ispu_bands(ax):
    ax.axhspan(0,   50,  alpha=0.09, color="#00c48a", zorder=0)
    ax.axhspan(50,  100, alpha=0.07, color="#ffd166", zorder=0)
    ax.axhspan(100, 200, alpha=0.06, color="#ff7b30", zorder=0)

def style_legend(ax):
    ax.legend(facecolor="#0b1829", edgecolor="#17395e",
              labelcolor="#ddeeff", fontsize=9)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈  Validasi Model", "🔮  Proyeksi Skenario", "🗂️  Data Mentah"])

# ── Tab 1 : Validasi ──────────────────────────────────────────────────────────
with tab1:
    pred = m.sim_monthly(p)

    fig, ax = dark_fig()
    add_ispu_bands(ax)
    ax.plot(obs.index, obs.values, "o", color="#48cae4",  ms=3.8,
            alpha=0.85, label="Observasi")
    ax.plot(pred.index, pred.values, "-", color="#ff6b6b", lw=2.0,
            label="Simulasi")
    ax.axhline(50,  color="#00c48a",  ls="--", lw=1.1, alpha=0.75,
               label="Batas Baik (50)")
    ax.axvline(pd.Timestamp("2019-01-01"), color="#ffd166", ls=":",
               lw=1.2, alpha=0.7, label="Mulai periode uji (2019)")
    ax.set_ylabel("PM10 (indeks ISPU)")
    style_legend(ax)
    st.pyplot(fig)
    plt.close()

    st.markdown("### 📊 Metrik Validasi")
    c1, c2 = st.columns(2)
    tr = m.metrics(obs.loc[:"2018-12-01"].iloc[12:], pred.loc[:"2018-12-01"].iloc[12:])
    te = m.metrics(obs.loc["2019-01-01":], pred.loc["2019-01-01":])
    with c1:
        st.markdown('<div class="info-box">📅 <b>Data Kalibrasi</b> — 2011 s/d 2018</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(tr, index=["nilai"]).round(3),
                     use_container_width=True)
    with c2:
        st.markdown('<div class="info-box">🔬 <b>Data Uji</b> — 2019 s/d 2025</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(te, index=["nilai"]).round(3),
                     use_container_width=True)

# ── Tab 2 : Skenario ──────────────────────────────────────────────────────────
with tab2:
    sims = {
        "Tanpa intervensi": m.simulate(p, end=f"{end_year}-12-31"),
        f"Pengurangan emisi {int(cut*100)}%": m.simulate(
            p, end=f"{end_year}-12-31", emis_cut=cut, cut_from=str(cut_from)),
    }
    SIM_COLORS = ["#ff6b6b", "#00b4d8"]

    fig, ax = dark_fig()
    add_ispu_bands(ax)
    rows = []
    for (name, sim), col in zip(sims.items(), SIM_COLORS):
        mm  = sim["C"].resample("MS").mean().loc["2024-01-01":]
        ax.plot(mm.index, mm.values, label=name, color=col, lw=2.1)
        fut = mm.loc[str(cut_from):]
        sh  = m.share_baik(fut.values, coef).mean() * 100
        rows.append({
            "Skenario": name,
            "Rata-rata PM10": round(fut.mean(), 1),
            "Est. Hari Baik (%)": round(sh, 1),
        })

    ax.axhline(50, color="#00c48a", ls="--", lw=1.1, alpha=0.75,
               label="Batas Baik (50)")
    ax.axvline(pd.Timestamp(str(cut_from)), color="#ffd166", ls=":",
               lw=1.2, alpha=0.7, label=f"Mulai kebijakan ({cut_from})")
    ax.set_ylabel("PM10 (indeks ISPU)")
    style_legend(ax)
    st.pyplot(fig)
    plt.close()

    st.markdown("### 📋 Ringkasan Skenario")
    st.table(pd.DataFrame(rows))
    st.markdown(
        '<div class="info-box">ℹ️ Proporsi hari baik diestimasi dari regresi logistik '
        'rata-rata bulanan terhadap proporsi hari PM10 ≤ 50.</div>',
        unsafe_allow_html=True,
    )

# ── Tab 3 : Data ─────────────────────────────────────────────────────────────
with tab3:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("### 🗂️ Data ISPU Harian — 500 Baris Terakhir")
        st.dataframe(d.reset_index().tail(500), use_container_width=True)
    with col_b:
        st.markdown("### 📉 Tren Bulanan PM10")
        st.line_chart(obs.dropna(), color="#00b4d8", use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Simulasi System Dynamics PM10 ISPU Jakarta &nbsp;·&nbsp;
  Stasiun DKI1 Bunderan HI &nbsp;·&nbsp; Data 2010–2025<br>
  Mata Kuliah Pemodelan dan Simulasi &nbsp;·&nbsp; Dibuat dengan Streamlit
</div>
""", unsafe_allow_html=True)
