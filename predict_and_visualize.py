import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial import cKDTree
from shapely import wkt
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

TARGET_COL = "height_B_median"
SPB_CENTER_X = 328_800
SPB_CENTER_Y = 6_641_000
N_SPATIAL_FOLDS = 5

FLOOR_BAND_ORDER = {
    "1": 0, "2": 1, "3-4": 2, "5": 3, "6-9": 4,
    "10-12": 5, "13-16": 5, "10-16": 5,
    "17-25": 6, "17+": 6, "26+": 7, "unknown": -1,
}
PURPOSE_MAP = {
    "Жилое": 0, "Жилой": 0, "Нежилое": 1, "Нежилой": 1,
    "Промышленное": 2, "Социальное": 3, "Торговое": 4,
    "Прочее": 5, "Неизвестно": -1,
}
CATEGORY_MAP = {
    "Жилое": 0, "Нежилое": 1, "Комплекс": 2,
    "Промышленное": 3, "Социальное": 4, "Постройка": 5,
}
HEIGHT_SRC_MAP = {
    "direct": 2, "height_consistent": 2,
    "calculated": 1, "height_calculated": 1, "missing": 0,
}


def safe_parse_wkt(s):
    try:
        if pd.isna(s) or s == "": return None
        g = wkt.loads(s)
        return g if g.is_valid else g.buffer(0)
    except Exception:
        return None


def geom_features(df):
    df = df.copy()
    df["area_m2"] = df["repr_area_m2"].astype(float)
    df["log_area"] = np.log1p(df["area_m2"])
    df["total_vertices"] = pd.to_numeric(df.get("geom_complexity", np.nan), errors="coerce")
    df["geom_parts"] = pd.to_numeric(df.get("geom_part_count", np.nan), errors="coerce").clip(lower=1)
    df["area_per_section"] = df["area_m2"] / df["geom_parts"].replace(0, np.nan)

    if "repr_geometry_wkt" not in df.columns:
        for c in ["perimeter_m", "log_perimeter", "npi", "bbox_length",
                  "bbox_width", "bbox_ratio", "bbox_fill_ratio", "convexity"]:
            df[c] = np.nan
        return df

    geoms = df["repr_geometry_wkt"].apply(safe_parse_wkt)
    per, npi_v, bl, bw, bf, cx = [], [], [], [], [], []
    for g, area in zip(geoms, df["area_m2"]):
        if g is None or area == 0:
            per.append(np.nan);
            npi_v.append(np.nan)
            bl.append(np.nan);
            bw.append(np.nan)
            bf.append(np.nan);
            cx.append(np.nan)
            continue
        try:
            p = g.length
            per.append(p)
            npi_v.append(2 * np.pi * np.sqrt(area) / p if p > 0 else np.nan)
            mrr = g.minimum_rotated_rectangle
            coords = list(mrr.exterior.coords)
            s1 = ((coords[1][0] - coords[0][0]) ** 2 + (coords[1][1] - coords[0][1]) ** 2) ** 0.5
            s2 = ((coords[2][0] - coords[1][0]) ** 2 + (coords[2][1] - coords[1][1]) ** 2) ** 0.5
            L, W = max(s1, s2), min(s1, s2)
            bl.append(L);
            bw.append(W)
            bf.append(area / (L * W) if L * W > 0 else np.nan)
            cx.append(area / g.convex_hull.area if g.convex_hull.area > 0 else np.nan)
        except Exception:
            per.append(np.nan);
            npi_v.append(np.nan)
            bl.append(np.nan);
            bw.append(np.nan)
            bf.append(np.nan);
            cx.append(np.nan)

    df["perimeter_m"] = per
    df["log_perimeter"] = np.log1p(df["perimeter_m"])
    df["npi"] = npi_v
    df["bbox_length"] = bl
    df["bbox_width"] = bw
    df["bbox_ratio"] = df["bbox_length"] / df["bbox_width"].replace(0, np.nan)
    df["bbox_fill_ratio"] = bf
    df["convexity"] = cx
    return df


def spatial_features(df, radii=(50, 100, 150, 200), main_r=100):
    df = df.copy()
    coords = df[["repr_centroid_x", "repr_centroid_y"]].values.astype(float)
    heights = df[TARGET_COL].values.astype(float) if TARGET_COL in df.columns else np.full(len(df), np.nan)
    tree = cKDTree(coords)

    df["centroid_x"] = coords[:, 0]
    df["centroid_y"] = coords[:, 1]
    df["dist_to_center"] = np.sqrt((coords[:, 0] - SPB_CENTER_X) ** 2 + (coords[:, 1] - SPB_CENTER_Y) ** 2)
    nn_dist, _ = tree.query(coords, k=2)
    df["nearest_neighbor_dist"] = nn_dist[:, 1]

    for r in radii:
        nbrs = tree.query_ball_point(coords, r=r)
        df[f"kde_{r}m"] = [max(0, len(i) - 1) / (np.pi * r ** 2) for i in nbrs]

    nbrs_main = tree.query_ball_point(coords, r=main_r)
    mean_h = np.full(len(df), np.nan);
    med_h = np.full(len(df), np.nan)
    p75_h = np.full(len(df), np.nan);
    std_h = np.full(len(df), np.nan)
    rng_h = np.full(len(df), np.nan);
    cnt_n = np.zeros(len(df), dtype=int)
    for i, idx in enumerate(nbrs_main):
        excl = [j for j in idx if j != i];
        cnt_n[i] = len(excl)
        h = heights[excl];
        h = h[~np.isnan(h)]
        if len(h) > 0:
            mean_h[i] = np.mean(h);
            med_h[i] = np.median(h)
            std_h[i] = np.std(h);
            rng_h[i] = np.max(h) - np.min(h)
        if len(h) >= 4:
            p75_h[i] = np.percentile(h, 75)

    df[f"mean_height_{main_r}m"] = mean_h
    df[f"median_height_{main_r}m"] = med_h
    df[f"p75_height_{main_r}m"] = p75_h
    df[f"std_height_{main_r}m"] = std_h
    df[f"height_range_{main_r}m"] = rng_h
    df[f"count_nbrs_{main_r}m"] = cnt_n

    zone_col = "district" if ("district" in df.columns and df["district"].notna().sum() > 10) else None
    if zone_col is None:
        df["_zone"] = pd.qcut(df["dist_to_center"], q=N_SPATIAL_FOLDS, labels=False, duplicates="drop")
        zone_col = "_zone"
    known = df[TARGET_COL].notna() if TARGET_COL in df.columns else pd.Series(False, index=df.index)
    zone_means = df[known].groupby(zone_col)[TARGET_COL].mean().rename("mean_height_by_zone")
    df = df.merge(zone_means, on=zone_col, how="left")
    if "_zone" in df.columns:
        df.drop(columns=["_zone"], inplace=True)
    return df


def categorical_features(df):
    df = df.copy()
    df["stairs_clean"] = pd.to_numeric(df.get("stairs", np.nan), errors="coerce").clip(1, 100)
    df["avg_floor_height"] = pd.to_numeric(df.get("avg_floor_height_clean", np.nan), errors="coerce").clip(2.0, 15.0)
    df["height_from_stairs"] = df["stairs_clean"] * df["avg_floor_height"]
    df["floor_mid_A"] = pd.to_numeric(df.get("floor_mid_A_median", np.nan), errors="coerce").clip(1, 100)

    if "floor_band" in df.columns:
        df["floor_band_enc"] = df["floor_band"].astype(str).map(FLOOR_BAND_ORDER).fillna(-1).astype(int)
    else:
        def _band(s):
            if pd.isna(s): return -1
            s = int(s)
            if s == 1:
                return 0
            elif s == 2:
                return 1
            elif s <= 4:
                return 2
            elif s == 5:
                return 3
            elif s <= 9:
                return 4
            elif s <= 16:
                return 5
            elif s <= 25:
                return 6
            else:
                return 7

        df["floor_band_enc"] = df["stairs_clean"].apply(_band)

    df["purpose_B_enc"] = (df["purpose_category_B_mode"].map(PURPOSE_MAP).fillna(-1).astype(int)
                           if "purpose_category_B_mode" in df.columns else -1)
    df["category_A_enc"] = (df["category_A_mode"].map(CATEGORY_MAP).fillna(-1).astype(int)
                            if "category_A_mode" in df.columns else -1)
    df["is_residential"] = (
            (df.get("purpose_category_B_mode", pd.Series([""] * len(df))) == "Жилое") |
            (df.get("category_A_mode", pd.Series([""] * len(df))) == "Жилое")
    ).astype(int)
    df["is_industrial_height"] = (df["avg_floor_height"].fillna(3.0) > 4.0).astype(int)
    df["height_source_enc"] = (df["height_source"].astype(str).str.lower().map(HEIGHT_SRC_MAP).fillna(0).astype(int)
                               if "height_source" in df.columns else 0)
    df["has_height"] = df[TARGET_COL].notna().astype(int) if TARGET_COL in df.columns else 0
    df["n_sources"] = ((pd.to_numeric(df.get("n_A", 0), errors="coerce").fillna(0) > 0).astype(int) +
                       (pd.to_numeric(df.get("n_B", 0), errors="coerce").fillna(0) > 0).astype(int))
    df["iou_ab_clean"] = pd.to_numeric(df.get("iou_ab", np.nan), errors="coerce").fillna(0.0)
    d_a = pd.to_numeric(df.get("detail_A", 0), errors="coerce").fillna(0)
    d_b = pd.to_numeric(df.get("detail_B", 0), errors="coerce").fillna(0)
    df["detail_ratio"] = d_a / (d_a + d_b).replace(0, np.nan)
    df["floor_height_ratio"] = df["avg_floor_height"].fillna(3.0) / 3.0
    df["area_floor_product"] = df["area_m2"] * df["stairs_clean"].fillna(1)
    df["log_area_floor_product"] = np.log1p(df["area_floor_product"])
    return df


# ── Графики ───────────────────────────────────────────────────────────────────

def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_distribution(y_pred, path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y_pred, bins=60, color="#27ae60", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(y_pred), color="red", lw=1.5, ls="--", label=f"медиана = {np.median(y_pred):.1f} м")
    ax.axvline(np.mean(y_pred), color="orange", lw=1.5, ls="--", label=f"среднее = {np.mean(y_pred):.1f} м")
    ax.set_xlabel("Предсказанная высота, м");
    ax.set_ylabel("Число зданий")
    ax.legend();
    plt.tight_layout();
    _savefig(fig, path)


def plot_diagnostics(y_true, y_pred, errors, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    lim = max(y_true.max(), y_pred.max()) * 1.05

    axes[0, 0].scatter(y_true, y_pred, s=4, alpha=0.2, color="#3498db", rasterized=True)
    axes[0, 0].plot([0, lim], [0, lim], "r--", lw=1.5)
    axes[0, 0].set_xlabel("Факт, м");
    axes[0, 0].set_ylabel("Прогноз, м")
    axes[0, 0].set_title(
        f"Predicted vs Actual  MAE={mean_absolute_error(y_true, y_pred):.2f} м  R²={r2_score(y_true, y_pred):.4f}")

    axes[0, 1].hist(errors, bins=80, color="#e74c3c", edgecolor="white", alpha=0.8)
    axes[0, 1].axvline(0, color="black", lw=1.5)
    axes[0, 1].axvline(np.median(errors), color="orange", lw=1.5, ls="--", label=f"median={np.median(errors):.2f} м")
    axes[0, 1].set_xlabel("Прогноз − Факт, м");
    axes[0, 1].legend()

    idx = np.random.choice(len(y_true), min(8000, len(y_true)), replace=False)
    axes[1, 0].scatter(y_true[idx], errors[idx], s=3, alpha=0.15, color="#2ecc71", rasterized=True)
    axes[1, 0].axhline(0, color="black", lw=1)
    bins = np.percentile(y_true, np.linspace(0, 100, 30))
    bin_idx = np.digitize(y_true, bins)
    bin_med = [np.median(errors[bin_idx == b]) if (bin_idx == b).sum() > 5 else np.nan for b in range(1, len(bins))]
    axes[1, 0].plot(bins[:-1], bin_med, "r-", lw=2, label="медиана ошибки")
    axes[1, 0].set_xlabel("Факт, м");
    axes[1, 0].set_ylabel("Ошибка, м");
    axes[1, 0].legend()

    scipy_stats.probplot(errors, plot=axes[1, 1])
    axes[1, 1].set_title("QQ-plot ошибок")
    plt.tight_layout();
    _savefig(fig, path)


def plot_mae_by_range(y_true, y_pred, path):
    bins = [0, 6, 12, 20, 30, 50, 200]
    labels = ["0–6", "6–12", "12–20", "20–30", "30–50", "50+"]
    maes, counts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_true >= lo) & (y_true < hi)
        maes.append(mean_absolute_error(y_true[m], y_pred[m]) if m.sum() > 0 else 0)
        counts.append(m.sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = np.arange(len(labels))
    bars = axes[0].bar(x, maes, color="#e74c3c", alpha=0.85)
    axes[0].set_xticks(x);
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("MAE, м");
    axes[0].set_title("MAE по диапазонам высот")
    for bar, v, cnt in zip(bars, maes, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{v:.2f}\nn={cnt:,}", ha="center", va="bottom", fontsize=8)
    axes[1].bar(x, counts, color="#3498db", alpha=0.85)
    axes[1].set_xticks(x);
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Число зданий");
    axes[1].set_title("Зданий по диапазонам")
    plt.tight_layout();
    _savefig(fig, path)


def plot_spatial(cx, cy, values, title, cbar_label, path, cmap="YlOrRd"):
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(cx, cy, c=values, s=4, alpha=0.5, cmap=cmap,
                    vmin=0, vmax=np.percentile(values, 95), rasterized=True)
    plt.colorbar(sc, ax=ax, label=cbar_label)
    ax.set_xlabel("UTM X, м");
    ax.set_ylabel("UTM Y, м");
    ax.set_title(title)
    plt.tight_layout();
    _savefig(fig, path)


def plot_feature_importance(feature_cols, importances, path, top_n=20):
    fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    ax.barh(fi.index[::-1], fi.values[::-1], color="#e67e22", alpha=0.85)
    ax.set_xlabel("MDI важность");
    ax.set_title(f"Feature Importance (топ-{top_n})")
    plt.tight_layout();
    _savefig(fig, path)


# ── Основная функция ──────────────────────────────────────────────────────────

def predict_and_visualize(data_path, model_path, meta_path, out_csv, plots_dir):
    plots = Path(plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    rf = joblib.load(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["features"]
    impute_medians = meta["impute_medians"]

    df = pd.read_csv(data_path, low_memory=False)
    df = geom_features(df)
    df = spatial_features(df)
    df = categorical_features(df)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = impute_medians.get(col, 0)
        elif col in impute_medians:
            df[col] = df[col].fillna(impute_medians[col])
    for fc in [c for c in feature_cols if c.endswith("_was_nan")]:
        base = fc.replace("_was_nan", "")
        if fc not in df.columns and base in df.columns:
            df[fc] = df[base].isna().astype(int)

    y_pred = np.clip(np.expm1(rf.predict(df[feature_cols].fillna(0).values)), 0, None)
    df["predicted_height_m"] = y_pred

    id_cols = [c for c in ["building_id", "src_a_id", "src_b_id"] if c in df.columns]
    coord_cols = [c for c in ["repr_centroid_x", "repr_centroid_y", "centroid_lon", "centroid_lat"] if c in df.columns]
    tgt_col = [TARGET_COL] if TARGET_COL in df.columns else []
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df[id_cols + coord_cols + tgt_col + ["predicted_height_m"]].to_csv(out_csv, index=False)

    plot_distribution(y_pred, plots / "01_prediction_distribution.png")

    if TARGET_COL in df.columns and df[TARGET_COL].notna().sum() > 10:
        mask = df[TARGET_COL].notna()
        y_true = df.loc[mask, TARGET_COL].values
        y_pred_known = y_pred[mask.values]
        errors = y_pred_known - y_true
        cx_col = next((c for c in ["repr_centroid_x", "centroid_x"] if c in df.columns), None)
        cy_col = next((c for c in ["repr_centroid_y", "centroid_y"] if c in df.columns), None)

        plot_diagnostics(y_true, y_pred_known, errors, plots / "02_diagnostics.png")
        plot_mae_by_range(y_true, y_pred_known, plots / "03_mae_by_range.png")
        if cx_col and cy_col:
            plot_spatial(df.loc[mask, cx_col].values, df.loc[mask, cy_col].values,
                         np.abs(errors), "Ошибки предсказания", "|ошибка|, м",
                         plots / "04_spatial_errors.png")

    if hasattr(rf, "feature_importances_"):
        plot_feature_importance(feature_cols, rf.feature_importances_, plots / "05_feature_importance.png")

    cx_col = next((c for c in ["repr_centroid_x", "centroid_x"] if c in df.columns), None)
    cy_col = next((c for c in ["repr_centroid_y", "centroid_y"] if c in df.columns), None)
    if cx_col and cy_col:
        plot_spatial(df[cx_col].values, df[cy_col].values, y_pred,
                     "Предсказанные высоты", "высота, м",
                     plots / "06_spatial_predictions.png", cmap="plasma")

    print(f"Готово. Предсказания: {out_csv} | Графики: {plots.resolve()}")


if __name__ == "__main__":
    data = sys.argv[1] if len(sys.argv) > 1 else "output/final_buildings_with_addr_and_new_features.csv"
    model = sys.argv[2] if len(sys.argv) > 2 else "output/rf_final_model.joblib"
    meta = sys.argv[3] if len(sys.argv) > 3 else "output/rf_final_model.json"
    out = sys.argv[4] if len(sys.argv) > 4 else "predictions/predictions_new.csv"
    plots = sys.argv[5] if len(sys.argv) > 5 else "plots"
    predict_and_visualize(data, model, meta, out, plots)
