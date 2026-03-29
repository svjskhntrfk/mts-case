import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt as swkt

warnings.filterwarnings("ignore")

AREA_MIN_M2 = 20.0
AREA_MAX_M2 = 100_000.0
GKH_FLOOR_MIN = 1
GKH_FLOOR_MAX = 100

SPB_LON_MIN, SPB_LON_MAX = 29.0, 31.5
SPB_LAT_MIN, SPB_LAT_MAX = 59.4, 60.4


def parse_tags(val) -> list:
    if pd.isna(val):
        return []
    try:
        result = ast.literal_eval(str(val).strip())
        if isinstance(result, list):
            return [str(t).strip() for t in result if str(t).strip()]
    except Exception:
        pass
    val = str(val).strip("[]").replace('"', '').replace("'", '')
    return [t.strip() for t in val.split(',') if t.strip()]


def categorize_tag(tag: str) -> str:
    t = str(tag).lower().strip()
    if any(x in t for x in ['жил', 'квартир']):                                return 'Жилое'
    if any(x in t for x in ['нежил', 'офис', 'коммерч']):                     return 'Нежилое'
    if 'комплекс' in t:                                                         return 'Комплекс'
    if any(x in t for x in ['промышл', 'склад', 'производ']):                 return 'Промышленное'
    if any(x in t for x in ['культур', 'образ', 'медиц', 'спорт', 'социал']): return 'Социальное'
    if any(x in t for x in ['постройк', 'сооруж', 'строен', 'гараж']):        return 'Постройка/сооружение'
    return 'Прочее'


def load_geom(wkt_str):
    try:
        g = swkt.loads(str(wkt_str))
        if not g.is_valid:
            g = g.buffer(0)
        return g if (g.is_valid and not g.is_empty) else None
    except Exception:
        return None


def geom_complexity(geom) -> int:
    if geom is None:                     return 0
    if geom.geom_type == 'Polygon':      return len(geom.exterior.coords)
    if geom.geom_type == 'MultiPolygon': return sum(len(p.exterior.coords) for p in geom.geoms)
    return 0


def clean_source_a(input_path: str, output_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['id'])
    df = df.drop_duplicates(subset=['geometry'])

    df['area_sq_m'] = pd.to_numeric(df['area_sq_m'], errors='coerce')
    df = df[df['area_sq_m'].notna() & df['area_sq_m'].between(AREA_MIN_M2, AREA_MAX_M2)].copy()

    df['tags_parsed'] = df['tags'].apply(parse_tags)
    df['tag_count'] = df['tags_parsed'].apply(len)
    df['primary_tag'] = df['tags_parsed'].apply(lambda x: x[0] if x else 'неизвестно')
    df['category'] = df['primary_tag'].apply(categorize_tag)
    df = df[~((df['category'] == 'Постройка/сооружение') & (df['area_sq_m'] < 50))].copy()

    for col in ['gkh_floor_count_min', 'gkh_floor_count_max']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col].notna() & ~df[col].between(GKH_FLOOR_MIN, GKH_FLOOR_MAX), col] = np.nan

    inv = (df['gkh_floor_count_min'].notna() & df['gkh_floor_count_max'].notna() &
           (df['gkh_floor_count_min'] > df['gkh_floor_count_max']))
    df.loc[inv, ['gkh_floor_count_min', 'gkh_floor_count_max']] = \
        df.loc[inv, ['gkh_floor_count_max', 'gkh_floor_count_min']].values

    df['has_gkh_floors'] = (df['gkh_floor_count_min'].notna() | df['gkh_floor_count_max'].notna()).astype(int)
    df['floor_range'] = np.where(
        df['gkh_floor_count_min'].notna() & df['gkh_floor_count_max'].notna(),
        df['gkh_floor_count_max'] - df['gkh_floor_count_min'], np.nan)
    df['floor_mid'] = np.where(
        df['gkh_floor_count_min'].notna() & df['gkh_floor_count_max'].notna(),
        (df['gkh_floor_count_min'] + df['gkh_floor_count_max']) / 2,
        df['gkh_floor_count_max'].fillna(df['gkh_floor_count_min']))

    df['has_gkh_address'] = df['gkh_address'].notna().astype(int)

    df['_geom'] = df['geometry'].apply(load_geom)
    df = df[df['_geom'].notna()].copy()

    df['centroid_lon'] = df['_geom'].apply(lambda g: round(g.centroid.x, 6))
    df['centroid_lat'] = df['_geom'].apply(lambda g: round(g.centroid.y, 6))
    df = df[df['centroid_lon'].between(SPB_LON_MIN, SPB_LON_MAX) &
            df['centroid_lat'].between(SPB_LAT_MIN, SPB_LAT_MAX)].copy()

    df['geom_type'] = df['_geom'].apply(lambda g: g.geom_type)
    df['geom_part_count'] = df['_geom'].apply(lambda g: len(list(g.geoms)) if g.geom_type == 'MultiPolygon' else 1)
    df['geom_complexity'] = df['_geom'].apply(geom_complexity)
    df['is_multipolygon'] = (df['geom_type'] == 'MultiPolygon').astype(int)
    df['geom_area_deg2'] = df['_geom'].apply(lambda g: round(g.area, 10))
    df['geom_area_approx_m2'] = df['geom_area_deg2'] * 111320 * 55660
    df['area_geom_ratio'] = df['area_sq_m'] / df['geom_area_approx_m2'].replace(0, np.nan)
    df['geometry_clean'] = df['_geom'].apply(lambda g: g.wkt)
    df = df.drop(columns=['_geom'])

    keep_cols = [
        'id', 'title',
        'geometry_clean', 'area_sq_m', 'geom_area_approx_m2', 'area_geom_ratio',
        'tags', 'tags_parsed', 'primary_tag', 'category', 'tag_count',
        'gkh_address', 'has_gkh_address',
        'gkh_floor_count_min', 'gkh_floor_count_max',
        'has_gkh_floors', 'floor_range', 'floor_mid',
        'geom_type', 'geom_part_count', 'geom_complexity', 'is_multipolygon',
        'centroid_lon', 'centroid_lat',
    ]
    df_clean = df[[c for c in keep_cols if c in df.columns]].copy()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)

    return df_clean


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "input/cup_it_example_src_A.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/cleaned_source_a.csv"
    clean_source_a(src, out)
