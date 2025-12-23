# make_corr_heatmap_annotated.py
# Figure 7: Correlation heatmap WITH annotated correlation coefficients (r)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Seaborn is the easiest for annotated heatmaps
try:
    import seaborn as sns
except ImportError:
    raise SystemExit("Seaborn not found. Install with: pip install seaborn")

# -------------------------
# USER SETTINGS
# -------------------------
DATA_PATH = "adapvtest.csv"
SEP = ";"  # your dataset uses ';'
OUT_PNG = "Figure7_corr_annotated.png"
OUT_PDF = "Figure7_corr_annotated.pdf"

# If you want to include the target (dver) in the correlation matrix, set True.
# If reviewer expects only feature-feature correlations, set False.
INCLUDE_DVER = True

# If you want to show only the lower triangle (cleaner), set True.
MASK_UPPER_TRIANGLE = False

# Bigger fonts / clearer for paper
FIGSIZE = (11, 9)
ANNOT_FONT_SIZE = 12
AXIS_FONT_SIZE = 13
TITLE_FONT_SIZE = 15

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(DATA_PATH, sep=SEP, engine="python")

features = ["pga", "H", "B", "q", "depth", "thickness"]
target = "dver"

cols = features + ([target] if INCLUDE_DVER else [])
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}\nAvailable columns: {list(df.columns)}")

# Correlation
corr = df[cols].corr(method="pearson")

# Optional: nicer axis labels for paper
pretty = {
    "pga": "PGA",
    "H": "H",
    "B": "B",
    "q": "q",
    "depth": "Dliq",
    "thickness": "Hliq",
    "dver": "Damage class (dver)",
}
corr = corr.rename(index=pretty, columns=pretty)

# -------------------------
# PLOT
# -------------------------
sns.set_theme(style="white")

plt.figure(figsize=FIGSIZE)

mask = None
if MASK_UPPER_TRIANGLE:
    mask = np.triu(np.ones_like(corr, dtype=bool))

ax = sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    square=True,
    linewidths=0.6,
    cbar_kws={"label": "Pearson correlation (r)", "shrink": 0.85},
    annot_kws={"size": ANNOT_FONT_SIZE},
)

# Make annotation text color readable on strong colors (optional but helpful)
# White text on |r|>0.55, otherwise black.
for t in ax.texts:
    try:
        val = float(t.get_text())
        t.set_color("white" if abs(val) > 0.55 else "black")
    except ValueError:
        pass

ax.set_title("Correlation matrix (Pearson r) with annotated coefficients", fontsize=TITLE_FONT_SIZE, pad=12)
ax.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.show()

print(f"Saved: {OUT_PNG}")
print(f"Saved: {OUT_PDF}")
