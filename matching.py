"""
build_final.py — матчинг A+B и сборка final_buildings_with_addr_and_new_features.csv

python build_final.py [path_a] [path_b] [output_path]
"""

import sys
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely import wkt as swkt
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

SOURCE_CRS = "EPSG:4326"
WORK_CRS   = "EPSG:32636"


# ── Геометрия ────────────────────────────────────────────────────────────────

def safe_parse_geom(x):
    if pd.isna(x): return None
    try:
        g = swkt.loads(str(x).strip())
        return g if g.is_valid else g.buffer(0)
    except Exception:
        return None

def union_or_none(geoms):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return unary_union(geoms) if geoms else None

def geom_complexity(geom):
    if geom is None: return 0
    if geom.geom_type == "Polygon":      return len(geom.exterior.coords)
    if geom.geom_type == "MultiPolygon": return sum(len(p.exterior.coords) for p in geom.geoms)
    return 0

def safe_iou(g1, g2):
    if g1 is None or g2 is None or g1.is_empty or g2.is_empty: return np.nan
    inter = g1.intersection(g2).area; union = g1.union(g2).area
    return inter / union if union > 0 else np.nan

def safe_support(src, other):
    if src is None or other is None or src.is_empty or other.is_empty: return np.nan
    inter = src.intersection(other).area; area = src.area
    return inter / area if area > 0 else np.nan


# ── Матчинг ───────────────────────────────────────────────────────────────────

def score_pairs(gA, gB, cand):
    a_geom = gA.set_index("a_idx").geometry
    b_geom = gB.set_index("b_idx").geometry
    rows = []
    for row in cand.itertuples(index=False):
        ga, gb     = a_geom.loc[row.a_idx], b_geom.loc[row.b_idx]
        inter_area = float(ga.intersection(gb).area)
        area_a, area_b = float(ga.area), float(gb.area)
        union_area = area_a + area_b - inter_area
        rows.append({
            "a_idx": int(row.a_idx), "b_idx": int(row.b_idx),
            "inter_area": inter_area, "area_a": area_a, "area_b": area_b,
            "iou":          inter_area / union_area if union_area > 0 else 0.0,
            "cover_a":      inter_area / area_a if area_a > 0 else 0.0,
            "cover_b":      inter_area / area_b if area_b > 0 else 0.0,
            "centroid_dist": float(ga.centroid.distance(gb.centroid)),
        })
    return pd.DataFrame(rows)


def make_edges(scores):
    s = scores.copy()
    s["size_ratio"]         = np.minimum(s["area_a"], s["area_b"]) / np.maximum(s["area_a"], s["area_b"])
    s["max_cover"]          = s[["cover_a","cover_b"]].max(axis=1)
    s["min_cover"]          = s[["cover_a","cover_b"]].min(axis=1)
    s["small_eq_radius"]    = np.sqrt(np.minimum(s["area_a"], s["area_b"]) / np.pi).clip(lower=1.0)
    s["centroid_dist_norm"] = s["centroid_dist"] / s["small_eq_radius"]
    s["score"] = (0.45*s["iou"] + 0.25*s["max_cover"] + 0.20*s["size_ratio"]
                  + 0.10*np.clip(1 - s["centroid_dist_norm"]/3.0, 0, 1))

    strong = (s["iou"] >= 0.18) | ((s["max_cover"] >= 0.85) & (s["size_ratio"] >= 0.20) & (s["centroid_dist_norm"] <= 2.0))
    weak   = (s["score"] >= 0.40) & (s["size_ratio"] >= 0.10) & (s["centroid_dist_norm"] <= 3.0)
    f = s[strong | weak].copy()
    f["rank_a"] = f.groupby("a_idx")["score"].rank(ascending=False, method="dense")
    f["rank_b"] = f.groupby("b_idx")["score"].rank(ascending=False, method="dense")
    return f[(f["rank_a"] <= 2) & (f["rank_b"] <= 2)].copy()


def parse_nodes(node_list):
    a_ids, b_ids = [], []
    for node in node_list:
        src, idx = node.split("_"); idx = int(idx)
        (a_ids if src == "A" else b_ids).append(idx)
    return a_ids, b_ids


def build_components(edges):
    G = nx.Graph()
    for i in edges["a_idx"].unique(): G.add_node(f"A_{i}")
    for j in edges["b_idx"].unique(): G.add_node(f"B_{j}")
    for row in edges.itertuples(index=False): G.add_edge(f"A_{row.a_idx}", f"B_{row.b_idx}")
    rows = []
    for cid, comp in enumerate(nx.connected_components(G)):
        a_nodes = [x for x in comp if x.startswith("A_")]
        b_nodes = [x for x in comp if x.startswith("B_")]
        n_a, n_b = len(a_nodes), len(b_nodes)
        rows.append({"cluster_id": cid, "n_A": n_a, "n_B": n_b,
                     "n_nodes": len(comp), "nodes": list(comp),
                     "comp_type": ("1x1" if n_a==1 and n_b==1 else
                                   "1xN" if n_a==1 and n_b>1 else
                                   "Nx1" if n_a>1 and n_b==1 else
                                   "NxN" if n_a>1 and n_b>1 else "single_source")})
    return pd.DataFrame(rows)


def is_suspicious(row, gA, gB):
    if row["n_A"] == 1 and row["n_B"] == 1: return False
    if row["n_nodes"] >= 4: return True
    a_ids, b_ids = parse_nodes(row["nodes"])
    geomA = union_or_none(gA[gA["a_idx"].isin(a_ids)].geometry.tolist())
    geomB = union_or_none(gB[gB["b_idx"].isin(b_ids)].geometry.tolist())
    iou = safe_iou(geomA, geomB)
    sA, sB = safe_support(geomA, geomB), safe_support(geomB, geomA)
    if pd.notna(iou) and iou < 0.35: return True
    if pd.notna(sA) and pd.notna(sB) and min(sA, sB) < 0.55: return True
    return False


def refine_component(row, edges, gA, gB, score_thr=0.55, iou_thr=0.20, cdn_thr=1.5, gap_thr=3.0):
    a_ids, b_ids = parse_nodes(row["nodes"])
    local = edges[edges["a_idx"].isin(a_ids) & edges["b_idx"].isin(b_ids)].copy()
    if len(local) == 0: return [row["nodes"]]

    local["rank_a_local"] = local.groupby("a_idx")["score"].rank(ascending=False, method="dense")
    local["rank_b_local"] = local.groupby("b_idx")["score"].rank(ascending=False, method="dense")
    local["mutual_best"]  = (local["rank_a_local"] == 1) & (local["rank_b_local"] == 1)
    keep = (local["mutual_best"]
            | ((local["score"] >= score_thr) & (local["iou"] >= iou_thr) & (local["centroid_dist_norm"] <= cdn_thr))
            | ((local["max_cover"] >= 0.85) & (local["size_ratio"] >= 0.60)))
    local = local[keep].copy()

    G = nx.Graph()
    for a in a_ids: G.add_node(f"A_{a}")
    for b in b_ids: G.add_node(f"B_{b}")
    for r in local.itertuples(index=False): G.add_edge(f"A_{r.a_idx}", f"B_{r.b_idx}")

    subA = gA[gA["a_idx"].isin(a_ids)].set_index("a_idx")
    subB = gB[gB["b_idx"].isin(b_ids)].set_index("b_idx")
    for lst, prefix, sub in [(list(subA.index),"A_",subA), (list(subB.index),"B_",subB)]:
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                g1, g2 = sub.loc[lst[i],"geometry"], sub.loc[lst[j],"geometry"]
                if g1.intersects(g2) or g1.touches(g2) or g1.distance(g2) <= gap_thr:
                    G.add_edge(f"{prefix}{lst[i]}", f"{prefix}{lst[j]}")

    return [sorted(list(c)) for c in nx.connected_components(G)]


def choose_repr_geom(row, gA, gB, margin=0.03):
    a_ids, b_ids = parse_nodes(row["nodes"])
    subA = gA[gA["a_idx"].isin(a_ids)]
    subB = gB[gB["b_idx"].isin(b_ids)]

    if len(subA) == 0:
        return {"repr_source": "B_only", "repr_geometry": union_or_none(subB.geometry.tolist()),
                "score_A": np.nan, "score_B": 1.0, "iou_ab": np.nan,
                "support_A": np.nan, "support_B": np.nan, "detail_A": np.nan, "detail_B": np.nan}
    if len(subB) == 0:
        return {"repr_source": "A_only", "repr_geometry": union_or_none(subA.geometry.tolist()),
                "score_A": 1.0, "score_B": np.nan, "iou_ab": np.nan,
                "support_A": np.nan, "support_B": np.nan, "detail_A": np.nan, "detail_B": np.nan}

    geomA = union_or_none(subA.geometry.tolist())
    geomB = union_or_none(subB.geometry.tolist())
    iou_ab = safe_iou(geomA, geomB)
    sA, sB = safe_support(geomA, geomB), safe_support(geomB, geomA)

    cA = subA["geom_complexity"].fillna(0).sum() if "geom_complexity" in subA.columns else geom_complexity(geomA)
    cB = subB["geom_complexity"].fillna(0).sum() if "geom_complexity" in subB.columns else geom_complexity(geomB)
    det_sum = cA + cB
    dA, dB = (cA/det_sum, cB/det_sum) if det_sum > 0 else (0.5, 0.5)

    fA, fB = max(len(subA)-1, 0), max(len(subB)-1, 0)
    fn = fA + fB + 1
    arA = subA["area_geom_ratio"].median() if "area_geom_ratio" in subA.columns else np.nan
    apA = min(abs(np.log(arA)), 2.0)/2.0 if pd.notna(arA) and arA > 0 else 0.0

    scoreA = 0.35*dA + 0.30*(sA if pd.notna(sA) else 0) + 0.20*(iou_ab if pd.notna(iou_ab) else 0) + 0.10*(1-fA/fn) + 0.05*(1-apA)
    scoreB = 0.35*dB + 0.30*(sB if pd.notna(sB) else 0) + 0.20*(iou_ab if pd.notna(iou_ab) else 0) + 0.10*(1-fB/fn)

    if scoreA > scoreB + margin:   repr_source, repr_geom = "A",     geomA
    elif scoreB > scoreA + margin: repr_source, repr_geom = "B",     geomB
    elif cA >= cB:                 repr_source, repr_geom = "A_tie", geomA
    else:                          repr_source, repr_geom = "B_tie", geomB

    return {"repr_source": repr_source, "repr_geometry": repr_geom,
            "score_A": scoreA, "score_B": scoreB, "iou_ab": iou_ab,
            "support_A": sA, "support_B": sB, "detail_A": dA, "detail_B": dB}


# ── Адреса ────────────────────────────────────────────────────────────────────

def normalize_addr(x):
    if pd.isna(x): return None
    x = str(x).lower().replace("ё", "е")
    for pat in [r"\bг\.\s*", r"\bгород\s+", r"\bул\.\s*", r"\bулица\s+",
                r"\bд\.\s*", r"\bдом\s+", r"\bлит\.\s*"]:
        x = re.sub(pat, "", x)
    x = re.sub(r"\bкорп\.\s*", "корпус ", x)
    return re.sub(r"\s+", " ", x).strip(" ,.-") or None

def most_common(values):
    values = [v for v in values if pd.notna(v) and str(v).strip()]
    return Counter(values).most_common(1)[0][0] if values else None

def split_ids(x):
    if pd.isna(x): return []
    return [s.strip() for s in str(x).split(",") if s.strip()]


# ── Агрегация признаков ───────────────────────────────────────────────────────

def median_num(vals):
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    return float(v.median()) if len(v) else np.nan

def max_num(vals):
    v = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    return float(v.max()) if len(v) else np.nan


def build_lookups(gA_df, gB_df):
    gA_df = gA_df.copy(); gB_df = gB_df.copy()
    gA_df["a_key"] = gA_df["id"].astype(str)
    gB_df = gB_df.reset_index(drop=True); gB_df["b_key"] = gB_df.index.astype(str)

    def lookup(df, key_col, col):
        return df.set_index(key_col)[col].to_dict() if col in df.columns else {}

    return {
        "a_addr":          lookup(gA_df, "a_key", "gkh_address"),
        "b_stairs":        lookup(gB_df, "b_key", "stairs"),
        "b_avg_fh":        lookup(gB_df, "b_key", "avg_floor_height_clean"),
        "b_floor_band":    lookup(gB_df, "b_key", "floor_band"),
        "b_height_valid":  lookup(gB_df, "b_key", "is_height_valid"),
        "b_height_source": lookup(gB_df, "b_key", "height_source"),
        "b_purpose":       lookup(gB_df, "b_key", "purpose_category"),
        "b_height":        lookup(gB_df, "b_key", "height"),
        "a_category":      lookup(gA_df, "a_key", "category"),
        "a_floor_mid":     lookup(gA_df, "a_key", "floor_mid"),
        "a_geom_cx":       lookup(gA_df, "a_key", "geom_complexity"),
        "a_geom_pc":       lookup(gA_df, "a_key", "geom_part_count"),
        "b_geom_cx":       lookup(gB_df, "b_key", "geom_complexity"),
        "b_geom_pc":       lookup(gB_df, "b_key", "geom_part_count"),
    }


# ── Главная функция ───────────────────────────────────────────────────────────

def build_final(path_a, path_b, output_path):
    print("Загрузка данных...")
    df_a = pd.read_csv(path_a, low_memory=False)
    df_b = pd.read_csv(path_b, low_memory=False)

    geom_col_a = next(c for c in ["geometry_clean", "geometry"] if c in df_a.columns)
    geom_col_b = next(c for c in ["geometry_clean", "wkt", "geometry"] if c in df_b.columns)

    df_a["geometry"] = df_a[geom_col_a].apply(safe_parse_geom)
    df_b["geometry"] = df_b[geom_col_b].apply(safe_parse_geom)
    if "id" not in df_a.columns: df_a["id"] = df_a.index.astype(str)
    if "id" not in df_b.columns: df_b["id"] = df_b.index.astype(str)

    gA = gpd.GeoDataFrame(df_a, geometry="geometry", crs=SOURCE_CRS)
    gB = gpd.GeoDataFrame(df_b, geometry="geometry", crs=SOURCE_CRS)
    gA = gA[gA.geometry.notnull() & ~gA.geometry.is_empty].to_crs(WORK_CRS).reset_index(drop=True)
    gB = gB[gB.geometry.notnull() & ~gB.geometry.is_empty].to_crs(WORK_CRS).reset_index(drop=True)
    gA["a_idx"] = gA.index; gA = gA[gA.geometry.area > 0].copy()
    gB["b_idx"] = gB.index; gB = gB[gB.geometry.area > 0].copy()
    print(f"A: {len(gA):,} | B: {len(gB):,}")

    print("Матчинг...")
    cand = gpd.sjoin(gA[["a_idx","geometry"]], gB[["b_idx","geometry"]],
                     how="inner", predicate="intersects").reset_index(drop=True)
    cand = cand[["a_idx","b_idx"]].drop_duplicates()
    scores = score_pairs(gA, gB, cand)
    edges  = make_edges(scores)
    print(f"Принято пар: {len(edges):,}")

    comp_df = build_components(edges)
    mixed   = comp_df[comp_df["comp_type"] != "single_source"].copy()

    print("Уточнение компонент...")
    refined_rows = []
    for row in mixed.itertuples(index=False):
        rd = row._asdict()
        subs = refine_component(rd, edges, gA, gB) if is_suspicious(rd, gA, gB) else [rd["nodes"]]
        for bid, nodes in enumerate(subs, start=len(refined_rows)):
            a_ids, b_ids = parse_nodes(nodes)
            n_a, n_b = len(a_ids), len(b_ids)
            refined_rows.append({
                "building_id": bid, "parent_cluster_id": rd["cluster_id"],
                "split_from_parent": int(len(subs) > 1),
                "n_A": n_a, "n_B": n_b, "n_nodes": len(nodes),
                "comp_type": ("1x1" if n_a==1 and n_b==1 else "1xN" if n_a==1
                              else "Nx1" if n_b==1 else "NxN"),
                "nodes": nodes,
            })
    refined_df = pd.DataFrame(refined_rows)
    refined_df["building_id"] = range(len(refined_df))

    print("Сборка признаков...")
    lookups = build_lookups(df_a, df_b)
    rows = []
    for row in refined_df.itertuples(index=False):
        rd = row._asdict()
        a_ids, b_ids = parse_nodes(rd["nodes"])
        subA = gA[gA["a_idx"].isin(a_ids)]
        subB = gB[gB["b_idx"].isin(b_ids)]
        rep  = choose_repr_geom(rd, gA, gB)
        geom = rep["repr_geometry"]
        centroid = geom.centroid if geom is not None else None

        a_keys = [df_a.loc[i,"id"] if i in df_a.index else str(i) for i in a_ids]
        b_keys = [str(i) for i in b_ids]

        rows.append({
            "building_id":          rd["building_id"],
            "parent_cluster_id":    rd["parent_cluster_id"],
            "split_from_parent":    rd["split_from_parent"],
            "comp_type":            rd["comp_type"],
            "n_A":                  rd["n_A"],
            "n_B":                  rd["n_B"],
            "a_source_ids":         ",".join(map(str, a_keys)),
            "b_source_ids":         ",".join(b_keys),
            "repr_source":          rep["repr_source"],
            "repr_geometry_wkt":    geom.wkt if geom is not None else None,
            "repr_area_m2":         geom.area if geom is not None else np.nan,
            "repr_centroid_x":      centroid.x if centroid else np.nan,
            "repr_centroid_y":      centroid.y if centroid else np.nan,
            "score_A":              rep["score_A"], "score_B":  rep["score_B"],
            "iou_ab":               rep["iou_ab"],
            "support_A":            rep["support_A"], "support_B": rep["support_B"],
            "detail_A":             rep["detail_A"],  "detail_B":  rep["detail_B"],
            # агрегаты B
            "height_B_median":      median_num([lookups["b_height"].get(k) for k in b_keys]),
            "stairs":               median_num([lookups["b_stairs"].get(k) for k in b_keys]),
            "avg_floor_height_clean": median_num([lookups["b_avg_fh"].get(k) for k in b_keys]),
            "floor_band":           most_common([lookups["b_floor_band"].get(k) for k in b_keys]),
            "is_height_valid":      most_common([lookups["b_height_valid"].get(k) for k in b_keys]),
            "height_source":        most_common([lookups["b_height_source"].get(k) for k in b_keys]),
            "purpose_category_B_mode": most_common([lookups["b_purpose"].get(k) for k in b_keys]),
            # агрегаты A
            "floor_mid_A_median":   median_num([lookups["a_floor_mid"].get(k) for k in a_keys]),
            "category_A_mode":      most_common([lookups["a_category"].get(k) for k in a_keys]),
            # адрес
            "addr_norm":            (most_common([normalize_addr(lookups["b_addr"].get(k)) for k in b_keys])
                                     if hasattr(lookups, "b_addr")
                                     else most_common([normalize_addr(lookups["a_addr"].get(k)) for k in a_keys])),
            # геометрическая детальность
            "geom_complexity":      (max_num([lookups["a_geom_cx"].get(k) for k in a_keys])
                                     if rep["repr_source"].startswith("A")
                                     else max_num([lookups["b_geom_cx"].get(k) for k in b_keys])),
            "geom_part_count":      (max_num([lookups["a_geom_pc"].get(k) for k in a_keys])
                                     if rep["repr_source"].startswith("A")
                                     else max_num([lookups["b_geom_pc"].get(k) for k in b_keys])),
        })

    final_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"Готово: {output_path} ({len(final_df):,} зданий)")
    return final_df


if __name__ == "__main__":
    path_a = sys.argv[1] if len(sys.argv) > 1 else "output/cleaned_a.csv"
    path_b = sys.argv[2] if len(sys.argv) > 2 else "output/cleaned_b.csv"
    out = sys.argv[3] if len(sys.argv) > 3 else "output/final_buildings_with_addr_and_new_features.csv"
    build_final(path_a, path_b, out)