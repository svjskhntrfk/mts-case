import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CITY_PATTERNS = [r"\bсанкт[-\s]?петербург\b", r"\bспб\b", r"\bпитер\b"]

STREET_TYPE_MAP = {
    "улица": "ул", "ул": "ул", "ул.": "ул",
    "проспект": "пр-кт", "пр-т": "пр-кт", "пр-кт.": "пр-кт", "пр-т.": "пр-кт", "пр": "пр-кт",
    "переулок": "пер", "пер.": "пер",
    "шоссе": "ш", "ш.": "ш",
    "площадь": "пл", "пл.": "пл",
    "набережная": "наб", "наб.": "наб", "наб": "наб",
    "проезд": "проезд", "пр-д": "проезд",
    "бульвар": "б-р", "б-р": "б-р",
}


def norm_str(x):
    if pd.isna(x):
        return None
    s = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", str(x).strip().lower().replace("ё", "е"))).strip()
    return s or None


def build_addr(city, st_type, st_name, number=None, fraction=None,
               housing=None, building=None, structure=None, letter=None):
    parts = []
    if city:                parts.append(f"г {city}")
    if st_type and st_name: parts.append(f"{st_type} {st_name}")
    if number and number != "0":
        h = str(number)
        if fraction and fraction != "0": h += f"/{fraction}"
        if housing and housing != "0": h += f" корп{housing}"
        if building and building != "0": h += f" к{building}"
        if structure and structure != "0": h += f" стр{structure}"
        if letter and letter != "0": h += f" лит{letter}"
        parts.append(h)
    return " ".join(parts) if parts else np.nan


def parse_row_b(row) -> str:
    st_raw = norm_str(row.get("type_street"))
    st_type = STREET_TYPE_MAP.get(st_raw, st_raw) if st_raw else None
    st_name = norm_str(row.get("name_street"))

    num_raw = norm_str(row.get("number"))
    housing = norm_str(row.get("housing"))
    building = norm_str(row.get("building"))
    fraction = norm_str(row.get("fraction"))
    letter = norm_str(row.get("letter"))

    base_house, house_letter = None, None
    if num_raw:
        m = re.match(r"^(\d+)([а-я])?$", num_raw)
        if m:
            base_house, house_letter = m.group(1), m.group(2)
        else:
            base_house = num_raw

    return build_addr("санкт петербург", st_type, st_name,
                      number=base_house, fraction=fraction,
                      housing=housing, building=building,
                      letter=letter or house_letter)


def normalize_addresses_b(input_path: str, output_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df["addr_norm"] = df.apply(parse_row_b, axis=1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "output/cleaned_source_b.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/cleaned_b.csv"
    normalize_addresses_b(src, out)
