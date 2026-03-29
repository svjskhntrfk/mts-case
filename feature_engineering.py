import warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkt

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
        excl = [j for j in idx if j != i]
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


def build_features(df):
    df = geom_features(df)
    df = spatial_features(df)
    df = categorical_features(df)
    return df
