"""
Model System Dynamics Konsentrasi PM10 (ISPU) DKI1 Bunderan HI, Jakarta.
Dipakai bersama oleh notebook Google Colab dan aplikasi web (Streamlit).

Struktur model (stock and flow):
  Stock  C  : konsentrasi PM10 (indeks ISPU) di udara
  Stock  K  : tingkat pengendalian emisi (kebijakan, dengan delay)
  Inflow    : Emisi(t)   = E0 * (1+g)^(t/365) * (1-K)
  Outflow   : Dispersi(t)= C / tau(t),  tau(t)= tau0*(1 + s*cos(2*pi*(doy-phi)/365.25))
  Loop B1   : C naik -> K naik -> Emisi turun -> C turun   (balancing, ada delay)
  Loop R1   : waktu -> pertumbuhan aktivitas (g) -> Emisi naik -> C naik (reinforcing)
"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

C_THR = 50.0     # batas ISPU kategori BAIK
TAU_K = 120.0    # delay kebijakan (hari), ditetapkan
PARAM_NAMES = ["E0", "g", "tau0", "s", "phi", "Kmax"]
BOUNDS = [(1.0, 40.0), (-0.01, 0.02), (1.0, 20.0), (0.0, 0.9), (0.0, 365.0), (0.0, 0.5)]


# ---------- Data ----------
def load_data(path):
    d = pd.read_csv(path, parse_dates=["tanggal"]).set_index("tanggal").sort_index()
    for c in ["pm25", "pm10", "so2", "co", "o3", "no2", "max"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.loc[d["categori"] == "TIDAK ADA DATA", ["max"]] = np.nan
    return d


def monthly_series(d, col="pm10", min_obs=15):
    g = d[col].resample("MS")
    m = g.mean()
    n = g.count()
    m[n < min_obs] = np.nan          # bulan dengan data terlalu sedikit dibuang
    full = pd.date_range("2010-01-01", "2025-02-01", freq="MS")
    return m.reindex(full)


# ---------- Model ----------
def simulate(params, start="2010-01-01", end="2025-02-28", c0=None, emis_cut=0.0, cut_from=None):
    """Simulasi harian (Euler, dt=1 hari). Mengembalikan DataFrame harian C, K, emisi."""
    E0, g, tau0, s, phi, Kmax = params
    idx = pd.date_range(start, end)
    n = len(idx)
    doy = idx.dayofyear.values
    t0 = pd.Timestamp("2010-01-01")
    t_years = (idx - t0).days.values / 365.0
    C = np.zeros(n); K = np.zeros(n); EM = np.zeros(n)
    C[0] = c0 if c0 is not None else 50.0
    cut_idx = None if cut_from is None else (idx >= pd.Timestamp(cut_from))
    for i in range(n - 1):
        tau = tau0 * (1 + s * np.cos(2 * np.pi * (doy[i] - phi) / 365.25))
        tau = max(tau, 0.5)
        extra = emis_cut if (cut_idx is not None and cut_idx[i]) else 0.0
        EM[i] = E0 * (1 + g) ** t_years[i] * (1 - K[i]) * (1 - extra)
        disp = C[i] / tau
        C[i + 1] = max(C[i] + EM[i] - disp, 0.0)
        target = Kmax * min(1.0, max(C[i] - C_THR, 0.0) / C_THR)
        K[i + 1] = K[i] + (target - K[i]) / TAU_K
    EM[-1] = EM[-2]
    return pd.DataFrame({"C": C, "K": K, "emisi": EM}, index=idx)


def sim_monthly(params, **kw):
    sim = simulate(params, **kw)
    return sim["C"].resample("MS").mean()


# ---------- Metrik ----------
def metrics(obs, pred):
    df = pd.concat([obs, pred], axis=1, keys=["o", "p"]).dropna()
    e = df.p - df.o
    rmse = float(np.sqrt((e ** 2).mean()))
    mae = float(e.abs().mean())
    mape = float((e.abs() / df.o).mean() * 100)
    ss_res = float((e ** 2).sum()); ss_tot = float(((df.o - df.o.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    r = float(np.corrcoef(df.o, df.p)[0, 1])
    return {"n": len(df), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "R2": r2, "r": r}


# ---------- Kalibrasi ----------
def calibrate(obs_monthly, train_end="2018-12-01", seed=42, maxiter=60):
    train = obs_monthly.loc[:train_end]

    def loss(p):
        pred = sim_monthly(p, end=pd.Timestamp(train_end) + pd.offsets.MonthEnd(0))
        df = pd.concat([train, pred], axis=1, keys=["o", "p"]).dropna()
        df = df.iloc[12:]                       # buang 12 bulan awal (spin-up)
        return float(np.sqrt(((df.o - df.p) ** 2).mean()))

    res = differential_evolution(loss, BOUNDS, seed=seed, maxiter=maxiter, popsize=12,
                                 tol=1e-6, polish=False)
    return dict(zip(PARAM_NAMES, res.x)), res.fun


# ---------- Hubungan rata-rata bulanan -> proporsi hari BAIK ----------
def fit_share_baik(d):
    x = d["pm10"].resample("MS").mean()
    y = (d["pm10"] <= C_THR).astype(float).where(d["pm10"].notna()).resample("MS").mean()
    df = pd.concat([x, y], axis=1, keys=["m", "y"]).dropna()
    # logistik y = 1/(1+exp(a*(m-b))) dengan regresi pada logit
    yy = df.y.astype(float).clip(0.02, 0.98).values
    z = np.log(yy / (1 - yy))
    A = np.vstack([df.m.astype(float).values, np.ones(len(df))]).T
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coef  # z = coef0*m + coef1

def share_baik(m, coef):
    return 1 / (1 + np.exp(-(coef[0] * np.asarray(m) + coef[1])))


# ---------- Skenario ----------
def run_scenarios(params, cuts=(0.0, 0.10, 0.20, 0.30), cut_from="2025-03-01", end="2030-12-31"):
    out = {}
    for c in cuts:
        out[c] = simulate(params, end=end, emis_cut=c, cut_from=cut_from)
    return out


# ---------- Uji integrasi: langkah waktu dibagi n_sub ----------
def simulate_substep(params, n_sub=2, start="2010-01-01", end="2025-02-28"):
    """Euler dengan dt = 1/n_sub hari (tanpa skenario), untuk uji kepekaan langkah waktu."""
    E0, g, tau0, s, phi, Kmax = params
    idx = pd.date_range(start, end)
    doy = idx.dayofyear.values
    t0 = pd.Timestamp("2010-01-01")
    t_years = (idx - t0).days.values / 365.0
    dt = 1.0 / n_sub
    C, K = 50.0, 0.0
    out = np.zeros(len(idx))
    for i in range(len(idx)):
        out[i] = C
        for _ in range(n_sub):
            tau = max(tau0 * (1 + s * np.cos(2 * np.pi * (doy[i] - phi) / 365.25)), 0.5)
            em = E0 * (1 + g) ** t_years[i] * (1 - K)
            Cn = max(C + dt * (em - C / tau), 0.0)
            target = Kmax * min(1.0, max(C - C_THR, 0.0) / C_THR)
            K = K + dt * (target - K) / TAU_K
            C = Cn
    return pd.Series(out, index=idx).resample("MS").mean()
