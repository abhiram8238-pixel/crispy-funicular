import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(page_title="Mental Health Dashboard", layout="wide")

# --------------------------
# Style
# --------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #e0e7ff, #f0fdf4);
}
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 700;
}
.glass-card {
    background: rgba(255, 255, 255, 0.55);
    border-radius: 18px;
    padding: 25px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}
.metric-card {
    background: linear-gradient(135deg, #a5b4fc, #c7d2fe);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    font-weight: bold;
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
}
.metric-value {
    font-size: 32px;
    color: #1e1b4b;
    font-weight: 900;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load Data
# --------------------------
DATA_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    csvs = list(DATA_DIR.glob("*.csv"))
    frames = []
    for f in csvs:
        df = pd.read_csv(f)
        df["source_file"] = f.name
        df.columns = (
            df.columns.str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

df = load_data()
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

# --------------------------
# Header
# --------------------------
st.markdown("<h1 style='text-align:center;color:#1e3a8a;'>Mental Health Analytics</h1>", unsafe_allow_html=True)
st.write("Sort. Filter. Analyze. Judge society responsibly.")

# --------------------------
# Sidebar
# --------------------------
st.sidebar.title("Filters")

source = st.sidebar.selectbox("Pick Dataset", ["All"] + sorted(df["source_file"].unique()))
view = st.sidebar.radio(
    "Choose a chart to display",
    ["Data Preview", "Bar Chart", "Trend Over Time", "Treatment Gap Pie", "Correlation"]
)

df_view = df.copy()
if source != "All":
    df_view = df_view[df_view["source_file"] == source]

if "year" in df_view.columns:
    yr_min, yr_max = int(df_view["year"].min()), int(df_view["year"].max())
    yr_rng = st.sidebar.slider("Year Range", yr_min, yr_max, (yr_min, yr_max))
    df_view = df_view[(df_view["year"] >= yr_rng[0]) & (df_view["year"] <= yr_rng[1])]

st.sidebar.markdown("-----")

# --------------------------
# KPIs
# --------------------------
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'>Rows<br><span class='metric-value'>{len(df_view)}</span></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'>Numeric Columns<br><span class='metric-value'>{len(numeric_cols)}</span></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'>Entities<br><span class='metric-value'>{df_view.get('entity', pd.Series()).nunique()}</span></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --------------------------
# Visuals
# --------------------------
if view == "Data Preview":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Sample of Filtered Data")
    st.dataframe(df_view.head(15), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif view == "Bar Chart":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    category = "entity" if "entity" in df_view.columns else df_view.columns[0]
    metric = st.selectbox("Metric", numeric_cols)
    top_n = st.slider("Top N", 3, 20, 10)
    srt = st.radio("Order", ["Highest First", "Lowest First"]) == "Highest First"
    agg = df_view.groupby(category)[metric].mean().sort_values(ascending=not srt).head(top_n)
    fig, ax = plt.subplots(figsize=(12,5))
    agg.plot(kind="bar", ax=ax)
    ax.set_title("Top Entities Ranked")
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

elif view == "Trend Over Time":
    if "year" not in df_view.columns:
        st.warning("This dataset does not contain a 'year' column.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        tm = st.selectbox("Trend Metric", numeric_cols)
        grp = df_view.groupby("year", as_index=False)[tm].mean()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(grp["year"], grp[tm], marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel(tm)
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

elif view == "Treatment Gap Pie":
    gap_col = next((c for c in df_view.columns if "gap" in c or "percent" in c), None)
    if not gap_col:
        st.warning("No gap column found.")
    else:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        category = "entity" if "entity" in df_view.columns else df_view.columns[0]
        pie_data = df_view.groupby(category)[gap_col].mean().nlargest(8)
        fig, ax = plt.subplots()
        ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

elif view == "Correlation":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    corr = df_view[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Bottom Insights
# --------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.subheader("Insights")
st.write("""
• High gaps growing over time = systems asleep.  
• Correlations = symptom buddies. Useful for predictions.  
• Outlier entities = interesting or broken.  
""")
st.markdown("</div>", unsafe_allow_html=True)
