import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from shapely import wkt as swkt

warnings.filterwarnings("ignore")

AREA_MIN_M2 = 20.0
AREA_MAX_M2 = 100_000.0
HEIGHT_MIN_M = 2.0
HEIGHT_MAX_M = 500.0
STAIRS_MIN = 1
STAIRS_MAX = 100
AVG_FLOOR_H_MIN = 1.5
AVG_FLOOR_H_MAX = 50.0
HEIGHT_CONSISTENCY_THRESHOLD = 0.40

SPB_LON_MIN, SPB_LON_MAX = 29.0, 31.5
SPB_LAT_MIN, SPB_LAT_MAX = 59.4, 60.4

PURPOSE_MAP = {
    'жилое': 'Жилое', 'нежилое': 'Нежилое', 'многоквартирный': 'Жилое',
    'индивидуальный': 'ИЖС', 'офис': 'Нежилое', 'торгов': 'Коммерческое',
    'склад': 'Складское', 'производ': 'Промышленное', 'промышл': 'Промышленное',
    'образован': 'Социальное', 'медицин': 'Социальное', 'культур': 'Социальное',
    'спортив': 'Социальное', 'гараж': 'Гараж', 'хозяйствен': 'Хозяйственное',
}


def categorize_purpose(val) -> str:
    if pd.isna(val):
        return 'Неизвестно'
    v = str(val).lower().strip()
    for key, cat in PURPOSE_MAP.items():
        if key in v:
            return cat
    return 'Прочее'


def floor_band(s) -> str:
    if pd.isna(s) or s <= 0: return 'unknown'
    s = float(s)
    if s == 1:  return '1'
    if s == 2:  return '2'
    if s <= 4:  return '3-4'
    if s <= 5:  return '5'
    if s <= 9:  return '6-9'
    if s <= 12: return '10-12'
    if s <= 16: return '13-16'
    if s <= 25: return '17-25'
    return '26+'


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


def clean_source_b(input_path: str, output_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)

    for col in ['stairs', 'avg_floor_height', 'height']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=['wkt'])

    bad_stairs = df['stairs'].notna() & (~df['stairs'].between(STAIRS_MIN, STAIRS_MAX) | (df['stairs'] < 1))
    df.loc[bad_stairs, 'stairs'] = np.nan
    df['is_stairs_valid'] = df['stairs'].notna().astype(int)

    bad_afh = df['avg_floor_height'].notna() & ~df['avg_floor_height'].between(AVG_FLOOR_H_MIN, AVG_FLOOR_H_MAX)
    df.loc[bad_afh, 'avg_floor_height'] = np.nan
    df['avg_floor_height_clean'] = df['avg_floor_height'].where(df['avg_floor_height'].between(2.5, 6.0))
    df['is_industrial_height'] = df['avg_floor_height'].between(6, 15, inclusive='right').astype(int)

    bad_h = df['height'].notna() & ~df['height'].between(HEIGHT_MIN_M, HEIGHT_MAX_M)
    df.loc[bad_h, 'height'] = np.nan
    df['is_height_valid'] = df['height'].notna().astype(int)

    afh = df['avg_floor_height'].fillna(3.0)
    df['height_from_stairs'] = np.where(df['stairs'].notna(), df['stairs'] * afh, np.nan)
    df['height_consistency_ratio'] = np.where(
        df['height'].notna() & df['height_from_stairs'].notna() & (df['height_from_stairs'] > 0),
        df['height'] / df['height_from_stairs'], np.nan)
    df['heights_inconsistent'] = (
            df['height_consistency_ratio'].notna() &
            ~df['height_consistency_ratio'].between(
                1 - HEIGHT_CONSISTENCY_THRESHOLD, 1 + HEIGHT_CONSISTENCY_THRESHOLD)
    ).astype(int)

    conditions = [
        df['height'].notna() & (df['heights_inconsistent'] == 0),
        df['height'].notna() & (df['heights_inconsistent'] == 1),
        df['height'].isna() & df['stairs'].notna(),
    ]
    df['height_source'] = np.select(conditions, ['height_consistent', 'height_inconsistent', 'stairs_only'], 'unknown')

    df['purpose_category'] = df['purpose_of_building'].apply(categorize_purpose)

    str_cols = ['subject', 'district', 'type', 'locality', 'type_street',
                'name_street', 'number', 'letter', 'fraction', 'housing', 'building']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})

    def build_addr_key(row):
        parts = []
        if pd.notna(row.get('name_street')): parts.append(str(row['name_street']).strip())
        if pd.notna(row.get('number')):      parts.append(str(row['number']).strip())
        if pd.notna(row.get('housing')) and str(row.get('housing', '')).strip() not in ('', 'nan'):
            parts.append(f"к{row['housing']}")
        if pd.notna(row.get('building')) and str(row.get('building', '')).strip() not in ('', 'nan'):
            parts.append(f"стр{row['building']}")
        return ' '.join(parts) if len(parts) >= 2 else np.nan

    df['addr_full'] = df.apply(build_addr_key, axis=1)
    df['has_full_address'] = df['addr_full'].notna().astype(int)

    df['_geom'] = df['wkt'].apply(load_geom)
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
    df = df[df['geom_area_approx_m2'].between(AREA_MIN_M2, AREA_MAX_M2)].copy()

    df['geometry_clean'] = df['_geom'].apply(lambda g: g.wkt)
    df = df.drop(columns=['_geom'])

    df['log_height'] = np.where(df['height'].notna() & (df['height'] > 0), np.log1p(df['height']), np.nan)
    df['floor_band'] = df['stairs'].apply(floor_band)
    df['height_per_floor'] = np.where(
        df['height'].notna() & df['stairs'].notna() & (df['stairs'] > 0),
        df['height'] / df['stairs'], np.nan)

    keep_cols = [
        'geometry_clean', 'geom_type', 'geom_part_count', 'geom_complexity',
        'is_multipolygon', 'centroid_lon', 'centroid_lat', 'geom_area_approx_m2',
        'subject', 'district', 'type', 'locality', 'type_street',
        'name_street', 'number', 'letter', 'fraction', 'housing', 'building',
        'addr_full', 'has_full_address',
        'purpose_of_building', 'purpose_category',
        'stairs', 'is_stairs_valid',
        'avg_floor_height', 'avg_floor_height_clean', 'is_industrial_height',
        'height', 'is_height_valid', 'height_from_stairs',
        'height_consistency_ratio', 'heights_inconsistent', 'height_source',
        'height_per_floor', 'log_height', 'floor_band',
    ]
    df_clean = df[[c for c in keep_cols if c in df.columns]].copy()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)

    return df_clean


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "input/cup_it_example_src_B.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/cleaned_source_b.csv"
    clean_source_b(src, out)
