import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import numpy as np
import os

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Inflation Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
<style>
    /* Main layout */
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

    /* Hide default streamlit elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Hero header */
    .hero-title {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 4px;
    }
    .hero-sub {
        font-size: 15px;
        color: #6b7280;
        margin-bottom: 0;
    }

    /* Section headers */
    .section-header {
        font-size: 17px;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #2563eb;
        padding-left: 10px;
        margin: 0 0 1rem 0;
    }

    /* Metric cards */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-card .value {
        font-size: 26px;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.2;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #9ca3af;
        margin-top: 2px;
    }

    /* Result highlight box */
    .result-highlight {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 16px;
        font-weight: 600;
        color: #1d4ed8;
        margin-top: 0.75rem;
    }

    /* Comparison summary cards */
    .range-card-a {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 10px;
        padding: 1rem;
    }
    .range-card-b {
        background: #fff1f2;
        border: 1px solid #fecdd3;
        border-radius: 10px;
        padding: 1rem;
    }
    .range-label {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .range-stat {
        font-size: 22px;
        font-weight: 700;
        margin: 2px 0;
    }
    .range-stat-label {
        font-size: 12px;
        color: #6b7280;
    }

    /* Divider */
    .custom-divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* Table styling */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Button */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-size: 14px;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ==============================
# MATPLOTLIB STYLE
# ==============================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.facecolor': '#fafafa',
    'figure.facecolor': 'white',
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


# ==============================
# LOAD MODEL + DATA
# ==============================
@st.cache_resource
def load_model():
    model_path = '../models/best_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

@st.cache_data
def load_data():
    data_path = '../data/processed/final_dataset.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Demo data if file not found
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=156, freq='MS')
        cpi = np.cumsum(np.random.normal(0.4, 0.9, 156)) + 120
        wpi = np.cumsum(np.random.normal(0.35, 1.0, 156)) + 115
        df = pd.DataFrame({'date': dates, 'cpi': cpi, 'wpi': wpi})

    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

model = load_model()
df = load_data()


# ==============================
# FEATURE ENGINEERING
# ==============================
def create_features(data):
    d = data.copy()
    d['month']    = d['date'].dt.month
    d['year']     = d['date'].dt.year
    d['quarter']  = d['date'].dt.quarter
    d['lag_1']    = d['cpi'].shift(1)
    d['lag_2']    = d['cpi'].shift(2)
    d['lag_3']    = d['cpi'].shift(3)
    d['lag_6']    = d['cpi'].shift(6)
    d['lag_12']   = d['cpi'].shift(12)
    d['rolling_mean_3'] = d['cpi'].rolling(3).mean()
    d['rolling_std_3']  = d['cpi'].rolling(3).std()
    d['rolling_mean_6'] = d['cpi'].rolling(6).mean()
    d['rolling_std_6']  = d['cpi'].rolling(6).std()
    d['cpi_pct_change'] = d['cpi'].pct_change() * 100
    d['cpi_diff']       = d['cpi'].diff()
    d['wpi_pct_change'] = d['wpi'].pct_change() * 100
    d['wpi_lag1']       = d['wpi'].shift(1)
    d['trend_index']    = range(len(d))
    d['cpi_wpi_diff']   = d['cpi'] - d['wpi']
    return d.dropna()

def safe_predict(mdl, X):
    """Align features safely whether or not model stores feature_names_in_"""
    try:
        return mdl.predict(X[mdl.feature_names_in_])
    except AttributeError:
        return mdl.predict(X)


# ==============================
# SHOCK DETECTION HELPER
# ==============================
def detect_shocks(series: pd.Series, contamination: float = 0.1):
    df_s = series.dropna().to_frame()
    iso  = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(df_s)
    result = pd.Series(0, index=series.index)
    result[df_s.index] = labels
    return result  # -1 = shock, 1 = normal


# ==============================
# HERO HEADER
# ==============================
st.markdown('<p class="hero-title">📊 Inflation Forecasting & Shock Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Analyze CPI trends, detect anomalies, compare models, and forecast future inflation</p>', unsafe_allow_html=True)
st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ==============================
# TOP METRICS
# ==============================
df['cpi_pct_change'] = df['cpi'].pct_change() * 100

last_cpi   = df['cpi'].iloc[-1]
last_inf   = df['cpi_pct_change'].iloc[-1]
shocks_all = detect_shocks(df['cpi_pct_change'])
shock_n    = (shocks_all == -1).sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Latest CPI</div>
        <div class="value">{last_cpi:.1f}</div>
        <div class="sub">Most recent reading</div>
    </div>""", unsafe_allow_html=True)

with c2:
    color = "#16a34a" if last_inf >= 0 else "#dc2626"
    arrow = "▲" if last_inf >= 0 else "▼"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Inflation Rate</div>
        <div class="value" style="color:{color}">{arrow} {abs(last_inf):.2f}%</div>
        <div class="sub">Month-over-month</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Shocks Detected</div>
        <div class="value" style="color:#dc2626">{shock_n}</div>
        <div class="sub">Out of {len(df)} months</div>
    </div>""", unsafe_allow_html=True)

with c4:
    date_range = f"{df['date'].iloc[0].strftime('%b %Y')} – {df['date'].iloc[-1].strftime('%b %Y')}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">Data Coverage</div>
        <div class="value" style="font-size:17px;padding-top:6px">{date_range}</div>
        <div class="sub">{len(df)} monthly observations</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ==============================
# SECTION 1: INFLATION TREND
# ==============================
st.markdown('<p class="section-header">📈 Inflation Trend</p>', unsafe_allow_html=True)

fig1, ax1 = plt.subplots(figsize=(11, 3.5))
ax1.fill_between(df['date'], df['cpi_pct_change'], alpha=0.15, color='#2563eb')
ax1.plot(df['date'], df['cpi_pct_change'], color='#2563eb', linewidth=1.8, label='Inflation %')
ax1.axhline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
ax1.set_ylabel('CPI % Change')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=30, ha='right')
ax1.legend(loc='upper left', fontsize=10)
plt.tight_layout()
st.pyplot(fig1, use_container_width=True)
plt.close(fig1)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ==============================
# SECTION 2: SHOCK DETECTION
# ==============================
st.markdown('<p class="section-header">🚨 Shock Detection</p>', unsafe_allow_html=True)

df_clean = df.dropna(subset=['cpi_pct_change']).copy()
df_clean['shock'] = detect_shocks(df_clean['cpi_pct_change'])

col_s1, col_s2 = st.columns([3, 1])
with col_s2:
    contamination = st.slider(
        "Sensitivity (contamination %)",
        min_value=5, max_value=25, value=10, step=1,
        help="Higher = more shocks flagged"
    ) / 100.0
    df_clean['shock'] = detect_shocks(df_clean['cpi_pct_change'], contamination)

shocks_df = df_clean[df_clean['shock'] == -1]

with col_s1:
    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.plot(df_clean['date'], df_clean['cpi_pct_change'],
             color='#2563eb', linewidth=1.6, label='Inflation %', zorder=2)
    ax2.scatter(shocks_df['date'], shocks_df['cpi_pct_change'],
                color='#dc2626', s=55, zorder=3, label=f'Shock ({len(shocks_df)})', edgecolors='white', linewidths=0.6)
    ax2.axhline(0, color='#94a3b8', linewidth=0.8, linestyle='--')
    ax2.set_ylabel('CPI % Change')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=30, ha='right')
    ax2.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ==============================
# SECTION 3: MODEL COMPARISON
# ==============================
st.markdown('<p class="section-header">🤖 Model Comparison</p>', unsafe_allow_html=True)

df_feat = create_features(df)
train_size = int(len(df_feat) * 0.8)
train = df_feat.iloc[:train_size]
test  = df_feat.iloc[train_size:]

drop_cols = ['date', 'cpi']
X_train = train.drop(columns=drop_cols)
y_train = train['cpi']
X_test  = test.drop(columns=drop_cols)
y_test  = test['cpi']

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred  = lr.predict(X_test)
rf_pred  = rf.predict(X_test)
lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_pred))
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_pred))

model_rows = [
    {'Model': 'Linear Regression', 'RMSE': round(lr_rmse, 4), 'Notes': 'Baseline'},
    {'Model': 'Random Forest',     'RMSE': round(rf_rmse, 4), 'Notes': 'Ensemble'},
]

if model is not None:
    try:
        xgb_pred = safe_predict(model, X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
        model_rows.append({'Model': 'XGBoost (loaded)', 'RMSE': round(xgb_rmse, 4), 'Notes': '★ Best'})
    except Exception as e:
        st.warning(f"XGBoost prediction failed: {e}")

results_df = pd.DataFrame(model_rows).sort_values('RMSE').reset_index(drop=True)

col_m1, col_m2 = st.columns([1, 2])
with col_m1:
    st.dataframe(results_df, use_container_width=True, hide_index=True)

with col_m2:
    fig3, ax3 = plt.subplots(figsize=(7, 3))
    colors = ['#93c5fd', '#3b82f6', '#1d4ed8'][:len(results_df)]
    bars = ax3.barh(results_df['Model'], results_df['RMSE'], color=colors, height=0.5, edgecolor='white')
    for bar, val in zip(bars, results_df['RMSE']):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=10, fontweight='500')
    ax3.set_xlabel('RMSE (lower is better)')
    ax3.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ==============================
# SECTION 4: SINGLE DATE PREDICTION
# ==============================
st.markdown('<p class="section-header">🔮 Predict Inflation for a Custom Date</p>', unsafe_allow_html=True)

col_p1, col_p2 = st.columns([1, 3])
with col_p1:
    pred_year  = st.number_input("Year",  min_value=2025, max_value=2035, value=2026)
    pred_month = st.number_input("Month", min_value=1,    max_value=12,   value=6)
    run_pred   = st.button("Predict CPI")

with col_p2:
    if run_pred:
        last_row = df_feat.iloc[-1:].copy()
        last_row['year']    = pred_year
        last_row['month']   = pred_month
        last_row['quarter'] = (pred_month - 1) // 3 + 1

        X_last = last_row.drop(columns=drop_cols)

        if model is not None:
            pred_val = safe_predict(model, X_last)[0]
            model_used = "XGBoost"
        else:
            pred_val = safe_predict(rf, X_last)[0]
            model_used = "Random Forest"

        st.markdown(f"""
        <div class="result-highlight">
            📈 Predicted CPI for {pred_month:02d}/{pred_year}: <strong>{pred_val:.2f}</strong>
            &nbsp;&nbsp;<span style="font-size:13px;font-weight:400;color:#6b7280">({model_used})</span>
        </div>""", unsafe_allow_html=True)

        # Chart: last 30 months + predicted point
        hist = df.tail(30)
        pred_date = pd.Timestamp(f'{pred_year}-{pred_month:02d}-01')

        fig4, ax4 = plt.subplots(figsize=(8, 3.5))
        ax4.plot(hist['date'], hist['cpi'], color='#2563eb', linewidth=1.8, label='Historical CPI')
        ax4.scatter([pred_date], [pred_val], color='#dc2626', s=120, zorder=5,
                    label=f'Predicted ({pred_month}/{pred_year})', edgecolors='white', linewidths=1)
        ax4.axvline(df['date'].iloc[-1], color='#9ca3af', linewidth=1, linestyle=':', alpha=0.8)
        ax4.set_ylabel('CPI')
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.xticks(rotation=30, ha='right')
        ax4.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close(fig4)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ==============================
# SECTION 5: DATE RANGE COMPARISON  ← NEW
# ==============================
st.markdown('<p class="section-header">📅 Date Range Comparison — Inflation & Shocks</p>', unsafe_allow_html=True)
st.caption("Select two date ranges to compare their inflation trends and shock patterns side by side.")

col_ra, col_rb = st.columns(2)

with col_ra:
    st.markdown("**🔵 Range A**")
    ra_start = st.date_input("From", value=pd.to_datetime("2012-01-01"), key="ra_s")
    ra_end   = st.date_input("To",   value=pd.to_datetime("2015-12-01"), key="ra_e")

with col_rb:
    st.markdown("**🔴 Range B**")
    rb_start = st.date_input("From", value=pd.to_datetime("2019-01-01"), key="rb_s")
    rb_end   = st.date_input("To",   value=pd.to_datetime("2022-12-01"), key="rb_e")

run_compare = st.button("Compare Ranges")

if run_compare:
    ra_mask = (df_clean['date'] >= pd.Timestamp(ra_start)) & (df_clean['date'] <= pd.Timestamp(ra_end))
    rb_mask = (df_clean['date'] >= pd.Timestamp(rb_start)) & (df_clean['date'] <= pd.Timestamp(rb_end))
    ra_df = df_clean[ra_mask].copy()
    rb_df = df_clean[rb_mask].copy()

    if ra_df.empty or rb_df.empty:
        st.error("One or both date ranges have no data. Please adjust your selections.")
    else:
        ra_shock_df = ra_df[ra_df['shock'] == -1]
        rb_shock_df = rb_df[rb_df['shock'] == -1]

        # ── Summary metrics ──
        st.markdown("#### Summary")
        ms1, ms2, ms3, ms4, ms5, ms6 = st.columns(6)
        def avg_inf(d): return d['cpi_pct_change'].mean()
        def max_inf(d): return d['cpi_pct_change'].max()

        with ms1:
            st.markdown(f"""<div class="range-card-a">
                <div class="range-label" style="color:#2563eb">Range A</div>
                <div class="range-stat" style="color:#1d4ed8">{avg_inf(ra_df):.2f}%</div>
                <div class="range-stat-label">Avg inflation</div>
            </div>""", unsafe_allow_html=True)
        with ms2:
            st.markdown(f"""<div class="range-card-a">
                <div class="range-label" style="color:#2563eb">Range A</div>
                <div class="range-stat" style="color:#1d4ed8">{max_inf(ra_df):.2f}%</div>
                <div class="range-stat-label">Peak inflation</div>
            </div>""", unsafe_allow_html=True)
        with ms3:
            st.markdown(f"""<div class="range-card-a">
                <div class="range-label" style="color:#2563eb">Range A</div>
                <div class="range-stat" style="color:#1d4ed8">{len(ra_shock_df)}</div>
                <div class="range-stat-label">Shocks detected</div>
            </div>""", unsafe_allow_html=True)
        with ms4:
            st.markdown(f"""<div class="range-card-b">
                <div class="range-label" style="color:#dc2626">Range B</div>
                <div class="range-stat" style="color:#b91c1c">{avg_inf(rb_df):.2f}%</div>
                <div class="range-stat-label">Avg inflation</div>
            </div>""", unsafe_allow_html=True)
        with ms5:
            st.markdown(f"""<div class="range-card-b">
                <div class="range-label" style="color:#dc2626">Range B</div>
                <div class="range-stat" style="color:#b91c1c">{max_inf(rb_df):.2f}%</div>
                <div class="range-stat-label">Peak inflation</div>
            </div>""", unsafe_allow_html=True)
        with ms6:
            st.markdown(f"""<div class="range-card-b">
                <div class="range-label" style="color:#dc2626">Range B</div>
                <div class="range-stat" style="color:#b91c1c">{len(rb_shock_df)}</div>
                <div class="range-stat-label">Shocks detected</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Plot 1: Overlaid inflation comparison ──
        st.markdown("#### Inflation trend comparison")
        fig5, ax5 = plt.subplots(figsize=(11, 3.8))
        ax5.plot(ra_df['date'], ra_df['cpi_pct_change'],
                 color='#2563eb', linewidth=2, label=f'Range A ({ra_start} → {ra_end})')
        ax5.fill_between(ra_df['date'], ra_df['cpi_pct_change'], alpha=0.1, color='#2563eb')
        ax5.plot(rb_df['date'], rb_df['cpi_pct_change'],
                 color='#dc2626', linewidth=2, linestyle='--', label=f'Range B ({rb_start} → {rb_end})')
        ax5.fill_between(rb_df['date'], rb_df['cpi_pct_change'], alpha=0.08, color='#dc2626')
        ax5.axhline(0, color='#94a3b8', linewidth=0.8, linestyle=':')
        ax5.set_ylabel('CPI % Change')
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha='right')
        ax5.legend(loc='upper left', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

        # ── Plot 2: Side-by-side shock detection ──
        st.markdown("#### Shock detection by range")
        fig6, (axA, axB) = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)

        axA.plot(ra_df['date'], ra_df['cpi_pct_change'],
                 color='#2563eb', linewidth=1.6, label='Inflation %')
        axA.scatter(ra_shock_df['date'], ra_shock_df['cpi_pct_change'],
                    color='#dc2626', s=60, zorder=4, label=f'{len(ra_shock_df)} shocks',
                    edgecolors='white', linewidths=0.7)
        axA.axhline(0, color='#94a3b8', linewidth=0.7, linestyle='--')
        axA.set_title(f'Range A  ({ra_start} → {ra_end})', fontsize=11, fontweight='500')
        axA.set_ylabel('CPI % Change')
        axA.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axA.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(axA.xaxis.get_majorticklabels(), rotation=30, ha='right')
        axA.legend(fontsize=10)

        axB.plot(rb_df['date'], rb_df['cpi_pct_change'],
                 color='#dc2626', linewidth=1.6, label='Inflation %')
        axB.scatter(rb_shock_df['date'], rb_shock_df['cpi_pct_change'],
                    color='#f59e0b', s=60, zorder=4, label=f'{len(rb_shock_df)} shocks',
                    edgecolors='white', linewidths=0.7)
        axB.axhline(0, color='#94a3b8', linewidth=0.7, linestyle='--')
        axB.set_title(f'Range B  ({rb_start} → {rb_end})', fontsize=11, fontweight='500')
        axB.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        axB.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(axB.xaxis.get_majorticklabels(), rotation=30, ha='right')
        axB.legend(fontsize=10)

        plt.tight_layout()
        st.pyplot(fig6, use_container_width=True)
        plt.close(fig6)

        # ── Plot 3: CPI level comparison ──
        st.markdown("#### Raw CPI level comparison")
        fig7, ax7 = plt.subplots(figsize=(11, 3.5))
        ax7.plot(ra_df['date'], ra_df['cpi'], color='#2563eb', linewidth=2,
                 label=f'Range A CPI', marker='o', markersize=3)
        ax7.plot(rb_df['date'], rb_df['cpi'], color='#dc2626', linewidth=2, linestyle='--',
                 label=f'Range B CPI', marker='s', markersize=3)
        ax7.set_ylabel('CPI')
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax7.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=30, ha='right')
        ax7.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig7, use_container_width=True)
        plt.close(fig7)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


# ==============================
# SECTION 6: DOWNLOAD
# ==============================
st.markdown('<p class="section-header">📥 Download Data</p>', unsafe_allow_html=True)

col_dl1, col_dl2 = st.columns([2, 3])
with col_dl1:
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download full dataset (CSV)",
        data=csv_data,
        file_name="inflation_data.csv",
        mime="text/csv"
    )
with col_dl2:
    st.caption(f"Includes {len(df)} rows · columns: {', '.join(df.columns.tolist())}")