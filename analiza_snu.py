"""
Raport nr 3 — Sleep Health and Daily Performance Dataset
Analiza korelacji, selekcja cech, grupowanie k-means

Wymagane biblioteki:
    pip install pandas matplotlib seaborn scikit-learn numpy

Użycie:
    python analiza_snu.py

Skrypt zakłada, że plik CSV z danymi znajduje się w tym samym folderze.
Zmień nazwę pliku w zmiennej CSV_PATH jeśli trzeba.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score

# ──────────────────────────────────────────────
# KONFIGURACJA
# ──────────────────────────────────────────────

CSV_PATH = "sleep_data.csv"   # <-- zmień na nazwę swojego pliku CSV

KPI = "sleep_quality_score"   # główny wskaźnik jakości snu

FEATURES = [                  # 8 wybranych cech do modelowania
    "stress_score",
    "sleep_duration_hrs",
    "deep_sleep_percentage",
    "rem_percentage",
    "cognitive_performance_score",
    "screen_time_before_bed_mins",
    "caffeine_mg_before_bed",
    "heart_rate_resting_bpm",
]

K_CLUSTERS = 4

COLORS = {
    0: "#1D9E75",   # Klaster 1 — Zdrowi śpiący (zielony)
    1: "#D85A30",   # Klaster 2 — Przepracowani (czerwony)
    2: "#BA7517",   # Klaster 3 — Nocni Marze (pomarańczowy)
    3: "#534AB7",   # Klaster 4 — Rekonwalescenci (fioletowy)
}

CLUSTER_NAMES = {
    0: "Zdrowi śpiący",
    1: "Przepracowani",
    2: "Nocni Marze",
    3: "Rekonwalescenci",
}

# ──────────────────────────────────────────────
# WCZYTANIE DANYCH
# ──────────────────────────────────────────────

print("=" * 60)
print("WCZYTYWANIE DANYCH")
print("=" * 60)

# Jeśli nie masz pliku, skrypt generuje przykładowe dane
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Wczytano plik: {CSV_PATH}")
except FileNotFoundError:
    print(f"Plik '{CSV_PATH}' nie znaleziony.")
    print("Generuję przykładowy zbiór 5000 rekordów do demonstracji...\n")
    np.random.seed(42)
    n = 5000
    stress      = np.random.uniform(1, 10, n)
    sleep_dur   = np.clip(8.5 - 0.4 * stress + np.random.normal(0, 0.8, n), 4, 10)
    deep_sleep  = np.clip(35 - 2 * stress + np.random.normal(0, 5, n), 5, 55)
    rem         = np.clip(28 - 1.5 * stress + np.random.normal(0, 4, n), 5, 45)
    screen      = np.clip(20 + 8 * stress + np.random.normal(0, 15, n), 0, 180)
    caffeine    = np.clip(30 + 12 * stress + np.random.normal(0, 20, n), 0, 300)
    cog_perf    = np.clip(80 - 4 * stress + np.random.normal(0, 8, n), 30, 100)
    hr          = np.clip(65 + 1.5 * stress + np.random.normal(0, 5, n), 50, 100)
    sleep_qual  = np.clip(
        10 - 0.82 * stress + 0.6 * (sleep_dur - 6) + 0.3 * (deep_sleep / 10)
        + np.random.normal(0, 0.5, n), 1, 10
    )
    df = pd.DataFrame({
        "stress_score": stress,
        "sleep_duration_hrs": sleep_dur,
        "deep_sleep_percentage": deep_sleep,
        "rem_percentage": rem,
        "cognitive_performance_score": cog_perf,
        "screen_time_before_bed_mins": screen,
        "caffeine_mg_before_bed": caffeine,
        "heart_rate_resting_bpm": hr,
        "sleep_quality_score": sleep_qual,
        "bmi": np.random.uniform(18, 35, n),
        "room_temperature_celsius": np.random.uniform(16, 26, n),
        "nap_duration_mins": np.random.uniform(0, 90, n),
        "occupation": np.random.choice(
            ["Menedżer","Nauczyciel","Programista","Prawnik","Pielęgniarka","Inżynier"], n
        ),
    })
    print(f"Wygenerowano {n} przykładowych rekordów.\n")

print(f"Rozmiar zbioru: {df.shape[0]:,} wierszy × {df.shape[1]} kolumn")
print(f"Brakujące wartości: {df.isnull().sum().sum()}\n")


# ──────────────────────────────────────────────
# WYKRES 1 — KORELACJA Z KPI
# ──────────────────────────────────────────────

print("=" * 60)
print("ANALIZA 1: Korelacja zmiennych z KPI")
print("=" * 60)

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if KPI in num_cols:
    num_cols.remove(KPI)

corr_with_kpi = (
    df[num_cols + [KPI]]
    .corr()[KPI]
    .drop(KPI)
    .sort_values(key=abs, ascending=False)
)

print("\nTop 10 korelacji z KPI:")
print(corr_with_kpi.head(10).to_string())

fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = ["#D85A30" if v < 0 else "#1D9E75" for v in corr_with_kpi.values]
bars = ax.barh(corr_with_kpi.index, corr_with_kpi.values, color=colors_bar, edgecolor="white", height=0.65)
ax.axvline(0, color="#444", linewidth=0.8)
ax.axvline(0.5,  color="#1D9E75", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(-0.5, color="#D85A30", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axvline(0.3,  color="#1D9E75", linewidth=0.5, linestyle=":", alpha=0.4)
ax.axvline(-0.3, color="#D85A30", linewidth=0.5, linestyle=":", alpha=0.4)

for bar, val in zip(bars, corr_with_kpi.values):
    ax.text(
        val + (0.01 if val >= 0 else -0.01),
        bar.get_y() + bar.get_height() / 2,
        f"{val:+.2f}",
        va="center", ha="left" if val >= 0 else "right",
        fontsize=9, fontweight="bold",
        color="#1D9E75" if val > 0 else "#D85A30"
    )

ax.set_xlabel("Współczynnik korelacji Pearsona r", fontsize=11)
ax.set_title(f"Korelacja atrybutów z KPI: {KPI}", fontsize=13, fontweight="bold", pad=15)
ax.set_xlim(-1.15, 1.15)
ax.grid(axis="x", alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)

pos_patch = mpatches.Patch(color="#1D9E75", label="Korelacja dodatnia")
neg_patch = mpatches.Patch(color="#D85A30", label="Korelacja ujemna")
ax.legend(handles=[pos_patch, neg_patch], loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig("wykres1_korelacja_kpi.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: wykres1_korelacja_kpi.png\n")


# ──────────────────────────────────────────────
# WYKRES 2 — MACIERZ KORELACJI (HEATMAPA)
# ──────────────────────────────────────────────

print("=" * 60)
print("ANALIZA 2: Macierz korelacji między kluczowymi zmiennymi")
print("=" * 60)

heatmap_cols = [c for c in FEATURES if c in df.columns] + [KPI]
corr_matrix = df[heatmap_cols].corr()

print("\nMacierz korelacji:")
print(corr_matrix.round(2).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask, k=1)] = True    # pokazuje całą macierz (bez maski)

cmap = sns.diverging_palette(10, 150, as_cmap=True)
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 9, "weight": "bold"},
    ax=ax,
    cbar_kws={"shrink": 0.8, "label": "r Pearsona"}
)

ax.set_title("Macierz korelacji — wybrane zmienne", fontsize=13, fontweight="bold", pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("wykres2_macierz_korelacji.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: wykres2_macierz_korelacji.png\n")


# ──────────────────────────────────────────────
# WYKRES 3 — WAŻNOŚĆ CECH (RANDOM FOREST)
# ──────────────────────────────────────────────

print("=" * 60)
print("ANALIZA 3: Selekcja cech — ważność wg Random Forest")
print("=" * 60)

available = [c for c in FEATURES if c in df.columns]
X = df[available].dropna()
y = df.loc[X.index, KPI]

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = (
    pd.Series(rf.feature_importances_, index=available)
    .sort_values(ascending=True)
)

print("\nWażność cech (Random Forest):")
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 200)
    print(f"  {feat:<40} {imp:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [
    "#1D9E75" if v >= 0.15 else
    "#185FA5" if v >= 0.08 else
    "#BA7517"
    for v in importances.values
]
bars = ax.barh(importances.index, importances.values, color=bar_colors, edgecolor="white", height=0.6)

for bar, val in zip(bars, importances.values):
    ax.text(
        val + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", fontsize=9, fontweight="bold"
    )

ax.set_xlabel("Ważność cechy (Feature Importance)", fontsize=11)
ax.set_title("Selekcja cech — ważność wg Random Forest\n(im wyżej i ciemniej — tym ważniejsza)", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="x", alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)

p1 = mpatches.Patch(color="#1D9E75", label="Bardzo ważna (≥0.15)")
p2 = mpatches.Patch(color="#185FA5", label="Ważna (0.08–0.15)")
p3 = mpatches.Patch(color="#BA7517", label="Mało ważna (<0.08)")
ax.legend(handles=[p1, p2, p3], fontsize=9)

plt.tight_layout()
plt.savefig("wykres3_waznosc_cech.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: wykres3_waznosc_cech.png\n")


# ──────────────────────────────────────────────
# WYKRES 4 — ELBOW METHOD (dobór k)
# ──────────────────────────────────────────────

print("=" * 60)
print("ANALIZA 4a: Dobór liczby klastrów — Elbow Method")
print("=" * 60)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[available].dropna())

inertias     = []
silhouettes  = []
k_range      = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels, sample_size=2000, random_state=42))
    print(f"  k={k}  WCSS={km.inertia_:,.1f}  Silhouette={silhouettes[-1]:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(k_range, inertias, "o-", color="#185FA5", linewidth=2, markersize=7)
ax1.axvline(K_CLUSTERS, color="#D85A30", linestyle="--", linewidth=1.5, label=f"Wybrane k={K_CLUSTERS}")
ax1.set_xlabel("Liczba klastrów k", fontsize=11)
ax1.set_ylabel("WCSS (inercja)", fontsize=11)
ax1.set_title("Elbow Method", fontsize=12, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)
ax1.spines[["top", "right"]].set_visible(False)

ax2.plot(k_range, silhouettes, "s-", color="#1D9E75", linewidth=2, markersize=7)
ax2.axvline(K_CLUSTERS, color="#D85A30", linestyle="--", linewidth=1.5, label=f"Wybrane k={K_CLUSTERS}")
ax2.set_xlabel("Liczba klastrów k", fontsize=11)
ax2.set_ylabel("Silhouette Score", fontsize=11)
ax2.set_title("Indeks sylwetki", fontsize=12, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("Dobór optymalnej liczby klastrów", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("wykres4a_elbow.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: wykres4a_elbow.png\n")


# ──────────────────────────────────────────────
# WYKRES 5 — K-MEANS: SCATTER + CENTRA
# ──────────────────────────────────────────────

print("=" * 60)
print(f"ANALIZA 4b: K-means (k={K_CLUSTERS})")
print("=" * 60)

km_final = KMeans(n_clusters=K_CLUSTERS, init="k-means++", n_init=10, random_state=42)
df_clean = df[available].dropna().copy()
df_clean["Klaster"] = km_final.fit_predict(X_scaled)

sil_final = silhouette_score(X_scaled, df_clean["Klaster"], sample_size=2000, random_state=42)
print(f"\nSilhouette Score (k={K_CLUSTERS}): {sil_final:.3f}")

counts = df_clean["Klaster"].value_counts().sort_index()
print("\nLiczebność klastrów:")
for k, n in counts.items():
    pct = n / len(df_clean) * 100
    print(f"  Klaster {k+1} ({CLUSTER_NAMES.get(k, '')}): {n:,} ({pct:.1f}%)")

print("\nCentra klastrów (oryginalna skala):")
centers_df = df_clean.groupby("Klaster")[available].mean().round(2)
print(centers_df.to_string())

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()
pairs = [
    ("stress_score", "sleep_duration_hrs"),
    ("deep_sleep_percentage", "rem_percentage"),
    ("screen_time_before_bed_mins", "caffeine_mg_before_bed"),
    ("stress_score", "cognitive_performance_score"),
]

for ax, (xcol, ycol) in zip(axes, pairs):
    if xcol not in df_clean.columns or ycol not in df_clean.columns:
        ax.set_visible(False)
        continue
    sample = df_clean.sample(min(1500, len(df_clean)), random_state=42)
    for k in range(K_CLUSTERS):
        subset = sample[sample["Klaster"] == k]
        ax.scatter(
            subset[xcol], subset[ycol],
            c=COLORS[k], alpha=0.35, s=12, label=f"K{k+1}: {CLUSTER_NAMES.get(k,'')}",
            edgecolors="none"
        )
    # zaznacz centra
    for k in range(K_CLUSTERS):
        cx = centers_df.loc[k, xcol] if xcol in centers_df.columns else None
        cy = centers_df.loc[k, ycol] if ycol in centers_df.columns else None
        if cx is not None and cy is not None:
            ax.scatter(cx, cy, c=COLORS[k], s=160, marker="*",
                       edgecolors="black", linewidths=0.8, zorder=5)
    ax.set_xlabel(xcol, fontsize=9)
    ax.set_ylabel(ycol, fontsize=9)
    ax.set_title(f"{xcol}  vs  {ycol}", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.5)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(f"K-means (k={K_CLUSTERS}) — rozkład klastrów\n(★ = centrum klastra)", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("wykres5_kmeans_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: wykres5_kmeans_scatter.png\n")


# ──────────────────────────────────────────────
# WYKRES 6 — PROFIL KLASTRÓW (RADAR / BAR)
# ──────────────────────────────────────────────

print("=" * 60)
print("ANALIZA 4c: Profil klastrów — średnie znormalizowane")
print("=" * 60)

# Znormalizowane centra klastrów
centers_scaled = pd.DataFrame(
    scaler.transform(centers_df[available]),
    columns=available,
    index=centers_df.index
)
print("\nZnormalizowane centra klastrów (0–1):")
print(centers_scaled.round(3).to_string())

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(available))
width = 0.2

for i, k in enumerate(range(K_CLUSTERS)):
    vals = centers_scaled.loc[k].values
    bars = ax.bar(
        x + i * width, vals, width,
        label=f"K{k+1}: {CLUSTER_NAMES.get(k,'')}",
        color=COLORS[k], alpha=0.85, edgecolor="white"
    )

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([c.replace("_", "\n") for c in available], fontsize=8)
ax.set_ylabel("Wartość znormalizowana (0–1)", fontsize=10)
ax.set_title("Profile klastrów — średnie znormalizowane cech\n(im wyżej, tym wyższa wartość cechy w danym klastrze)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper right")
ax.set_ylim(0, 1.1)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("wykres6_profil_klastrow.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: wykres6_profil_klastrow.png\n")


# ──────────────────────────────────────────────
# WYKRES 7 — LICZEBNOŚĆ KLASTRÓW (DONUT)
# ──────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

labels_pie = [f"K{k+1}: {CLUSTER_NAMES.get(k,'')}\n{counts[k]/len(df_clean)*100:.1f}%" for k in range(K_CLUSTERS)]
colors_pie = [COLORS[k] for k in range(K_CLUSTERS)]
wedges, texts = ax1.pie(
    [counts[k] for k in range(K_CLUSTERS)],
    labels=None,
    colors=colors_pie,
    startangle=90,
    wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2}
)
ax1.legend(wedges, labels_pie, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=9)
ax1.set_title("Rozkład klastrów", fontsize=12, fontweight="bold")

# srednia jakos snu per klaster
if KPI in df.columns:
    df_clean_kpi = df_clean.copy()
    df_clean_kpi[KPI] = df.loc[df_clean.index, KPI]
    mean_kpi = df_clean_kpi.groupby("Klaster")[KPI].mean()
    bar_colors2 = [COLORS[k] for k in mean_kpi.index]
    bars2 = ax2.bar(
        [f"K{k+1}\n{CLUSTER_NAMES.get(k,'')}" for k in mean_kpi.index],
        mean_kpi.values,
        color=bar_colors2, edgecolor="white"
    )
    for bar, val in zip(bars2, mean_kpi.values):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                 f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel(f"Średni {KPI}", fontsize=10)
    ax2.set_title(f"Średnia jakość snu wg klastra", fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(mean_kpi.values) * 1.15)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("Podsumowanie klastrów k-means", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("wykres7_podsumowanie_klastrow.png", dpi=150, bbox_inches="tight")
plt.show()
print("Zapisano: wykres7_podsumowanie_klastrow.png\n")


# ──────────────────────────────────────────────
# PODSUMOWANIE KOŃCOWE
# ──────────────────────────────────────────────

print("=" * 60)
print("PODSUMOWANIE")
print("=" * 60)
print(f"\nNajsilniejszy prediktor:  {corr_with_kpi.index[0]}  (r = {corr_with_kpi.iloc[0]:+.2f})")
print(f"Najsłabszy prediktor:     {corr_with_kpi.index[-1]}  (r = {corr_with_kpi.iloc[-1]:+.2f})")
print(f"\nLiczba klastrów:          k = {K_CLUSTERS}")
print(f"Silhouette Score:         {sil_final:.3f}")
print(f"\nWygenerowane pliki:")
for i in range(1, 8):
    names = {
        1: "wykres1_korelacja_kpi.png       — korelacja zmiennych z KPI",
        2: "wykres2_macierz_korelacji.png   — heatmapa macierzy korelacji",
        3: "wykres3_waznosc_cech.png        — ważność cech (Random Forest)",
        4: "wykres4a_elbow.png              — dobór k (Elbow + Silhouette)",
        5: "wykres5_kmeans_scatter.png      — scatter plot klastrów",
        6: "wykres6_profil_klastrow.png     — profil klastrów (bar chart)",
        7: "wykres7_podsumowanie_klastrow.png — liczebność i KPI per klaster",
    }
    print(f"  {names[i]}")
print("\nGotowe!")
