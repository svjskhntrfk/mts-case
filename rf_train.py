import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy import stats as scipy_stats
from scipy.spatial import cKDTree
from shapely import wkt
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ── Константы ─────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TARGET_COL = "height_B_median"
SPB_CENTER_X = 328_800
SPB_CENTER_Y = 6_641_000
N_SPATIAL_FOLDS = 5
NULL_DROP_THRESH = 0.80
MISS_FLAG_THRESH = 0.05
LEAKAGE_CORR_DROP = 0.95
COLLINEAR_THRESH = 0.95
AUG_HEIGHT_THRESH = 50
AUG_N_COPIES = 3
AUG_NOISE_RANGE = 0.10

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "output/final_buildings_with_addr_and_new_features.csv"
MODEL_SAVE_PATH = sys.argv[2] if len(sys.argv) > 2 else "output/rf_final_model.joblib"
PREDICTIONS_PATH = sys.argv[3] if len(sys.argv) > 3 else "predictions/predictions_all_buildings.csv"

np.random.seed(RANDOM_STATE)

FLOOR_BAND_ORDER = {"1": 0, "2": 1, "3-4": 2, "5": 3, "6-9": 4, "10-12": 5, "13-16": 5,
                    "10-16": 5, "17-25": 6, "17+": 6, "26+": 7, "unknown": -1}
PURPOSE_MAP = {"Жилое": 0, "Жилой": 0, "Нежилое": 1, "Нежилой": 1, "Промышленное": 2,
               "Социальное": 3, "Торговое": 4, "Прочее": 5, "Неизвестно": -1}
CATEGORY_MAP = {"Жилое": 0, "Нежилое": 1, "Комплекс": 2, "Промышленное": 3, "Социальное": 4, "Постройка": 5}
HEIGHT_SRC_MAP = {"direct": 2, "height_consistent": 2, "calculated": 1, "height_calculated": 1, "missing": 0}

RF_PARAMS = dict(n_estimators=400, max_features="sqrt", min_samples_leaf=5,
                 max_depth=None, n_jobs=-1, random_state=RANDOM_STATE, oob_score=True)

ALL_CANDIDATES = [
    "area_m2", "log_area", "perimeter_m", "npi", "bbox_length", "bbox_width",
    "bbox_ratio", "bbox_fill_ratio", "total_vertices", "geom_parts",
    "centroid_x", "centroid_y", "dist_to_center", "nearest_neighbor_dist",
    "mean_height_100m", "median_height_100m", "p75_height_100m",
    "std_height_100m", "height_range_100m", "count_nbrs_100m",
    "kde_50m", "kde_100m", "kde_150m", "kde_200m", "mean_height_by_zone",
    "avg_floor_height", "floor_mid_A", "floor_band_enc",
    "purpose_B_enc", "category_A_enc", "is_residential", "is_industrial_height",
    "height_source_enc", "has_height", "n_sources", "iou_ab_clean", "detail_ratio",
    "area_floor_product", "log_area_floor_product", "stairs_clean", "height_from_stairs",
    "floor_height_ratio", "area_per_section", "log_perimeter", "convexity",
]

INTERPRETABILITY_ORDER = [
    "area_m2", "log_area", "area_floor_product", "log_area_floor_product",
    "mean_height_100m", "std_height_100m", "count_nbrs_100m", "kde_150m",
    "avg_floor_height", "bbox_fill_ratio", "floor_height_ratio", "area_per_section",
    "log_perimeter", "convexity", "median_height_100m", "height_range_100m", "kde_100m", "kde_200m",
]

GEO_AUG_FEATS = {
    "area_m2", "log_area", "perimeter_m", "npi", "bbox_length", "bbox_width",
    "bbox_ratio", "bbox_fill_ratio", "total_vertices", "geom_parts",
    "area_floor_product", "log_area_floor_product",
}


# ── Feature Engineering ───────────────────────────────────────────────────

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
        for c in ["perimeter_m", "log_perimeter", "npi", "bbox_length", "bbox_width",
                  "bbox_ratio", "bbox_fill_ratio", "convexity"]:
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
    heights = df[TARGET_COL].values.astype(float)
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
    zone_means = df[df[TARGET_COL].notna()].groupby(zone_col)[TARGET_COL].mean().rename("mean_height_by_zone")
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
    df["has_height"] = df[TARGET_COL].notna().astype(int)
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


# ── Pipeline ──────────────────────────────────────────────────────────────

df_raw = pd.read_csv(DATA_PATH)

df = df_raw.copy()
df = geom_features(df)
df = spatial_features(df)
df = categorical_features(df)
df["log_target"] = np.log1p(df[TARGET_COL])

df_train = df[df[TARGET_COL].notna()].copy()
target_vals = df_train[TARGET_COL].values

# фильтр утечки
leakage_drop = [
    f for f in ALL_CANDIDATES
    if f in df_train.columns and
       abs(scipy_stats.pearsonr(df_train[f].fillna(df_train[f].median()), target_vals)[0]) > LEAKAGE_CORR_DROP
]

# фильтр коллинеарности
cands = list(dict.fromkeys(f for f in ALL_CANDIDATES if f in df.columns and f not in leakage_drop))
corr_matrix = df_train[cands].fillna(df_train[cands].median()).corr().abs()
collinear_drop = set()
for i, f1 in enumerate(cands):
    for f2 in cands[i + 1:]:
        if corr_matrix.at[f1, f2] > COLLINEAR_THRESH:
            r1 = INTERPRETABILITY_ORDER.index(f1) if f1 in INTERPRETABILITY_ORDER else 999
            r2 = INTERPRETABILITY_ORDER.index(f2) if f2 in INTERPRETABILITY_ORDER else 999
            collinear_drop.add(f2 if r1 <= r2 else f1)

exclude = set(leakage_drop) | collinear_drop
null_pcts = df_train[cands].isna().mean()
FEATURE_COLS = [f for f in ALL_CANDIDATES
                if f in df.columns and f not in exclude and null_pcts.get(f, 0) <= NULL_DROP_THRESH]

# импутация
impute_medians = {col: float(df_train[col].median())
                  for col in FEATURE_COLS if df_train[col].isna().any()}

flag_cols = []
for col, med in impute_medians.items():
    if df_train[col].isna().mean() > MISS_FLAG_THRESH:
        flag = f"{col}_was_nan"
        df[flag] = df[col].isna().astype(int)
        flag_cols.append(flag)
    df[col] = df[col].fillna(med)

FINAL_FEATURES = FEATURE_COLS + flag_cols

# аугментация высоких зданий
df_train_all = df[df[TARGET_COL].notna()].copy()
df_predict = df[df[TARGET_COL].isna()].copy()

aug_feats = [f for f in FINAL_FEATURES if f in GEO_AUG_FEATS]
aug_parts = []
for _ in range(AUG_N_COPIES):
    chunk = df_train_all[df_train_all[TARGET_COL] > AUG_HEIGHT_THRESH].copy()
    for feat in aug_feats:
        chunk[feat] *= np.random.uniform(1 - AUG_NOISE_RANGE, 1 + AUG_NOISE_RANGE, size=len(chunk))
    chunk["log_target"] = np.log1p(chunk[TARGET_COL])
    aug_parts.append(chunk)

df_final_train = pd.concat([df_train_all] + aug_parts, ignore_index=True)

# обучение
X = df_final_train[FINAL_FEATURES].values
y = df_final_train["log_target"].values
h = df_final_train[TARGET_COL].values
w = np.where(h > 100, 8.0, np.where(h > 50, 5.0, np.where(h > 30, 2.0, 1.0)))

rf = RandomForestRegressor(**RF_PARAMS)
rf.fit(X, y, sample_weight=w)

# сохранение
joblib.dump(rf, MODEL_SAVE_PATH)
meta_path = Path(MODEL_SAVE_PATH).with_suffix(".json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump({"features": FINAL_FEATURES, "impute_medians": impute_medians,
               "oob_r2": round(rf.oob_score_, 4)}, f, ensure_ascii=False, indent=2)

# предсказание
if len(df_predict) > 0:
    for col in FINAL_FEATURES:
        if col not in df_predict.columns:
            df_predict[col] = impute_medians.get(col, 0)
    df_predict = df_predict.copy()
    df_predict["predicted_height_m"] = np.clip(np.expm1(rf.predict(df_predict[FINAL_FEATURES].fillna(0).values)), 0,
                                               None)
    id_cols = [c for c in ["building_id", "src_a_id", "src_b_id"] if c in df_predict.columns]
    Path(PREDICTIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    df_predict[id_cols + ["predicted_height_m"]].to_csv(PREDICTIONS_PATH, index=False)
