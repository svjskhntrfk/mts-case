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

HOUSE_WORDS = [r"\bд\b", r"\bдом\b", r"\bдома\b"]
BUILDING_WORDS = [r"\bк\b", r"\bкорп\b", r"\bкорпус\b"]
STRUCT_WORDS = [r"\bстр\b", r"\bстроен\b", r"\bстроение\b"]
LETTER_WORDS = [r"\bлит\b", r"\bлитера\b"]


def clean_text(val) -> str:
    if pd.isna(val):
        return ""
    s = str(val).lower().replace("ё", "е")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[,\.;]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_city(s: str):
    for pat in CITY_PATTERNS:
        if re.search(pat, s):
            return "санкт петербург", re.sub(r"\s+", " ", re.sub(pat, " ", s)).strip()
    return "санкт петербург", s


def extract_street(s: str):
    tokens = s.split()
    idx = next((i for i, t in enumerate(tokens) if t in STREET_TYPE_MAP), None)
    if idx is None:
        return None, None, s

    st_type = STREET_TYPE_MAP[tokens[idx]]
    end = len(tokens)
    for j in range(idx + 1, len(tokens)):
        if re.match(r"^\d+([\/\-]\d+)?[а-я]?$", tokens[j]):
            end = j;
            break
        if any(re.match(p, tokens[j]) for p in HOUSE_WORDS + BUILDING_WORDS + STRUCT_WORDS + LETTER_WORDS):
            end = j;
            break

    st_name = " ".join(tokens[idx + 1:end]).strip() or None
    rest = " ".join(tokens[:idx] + tokens[end:]).strip()
    return st_type, st_name, rest


def extract_by_words(s: str, patterns):
    tokens = s.split()
    for i in range(len(tokens) - 1):
        if any(re.match(p, tokens[i]) for p in patterns):
            return tokens[i + 1], " ".join(tokens[:i] + tokens[i + 2:]).strip()
    for i, t in enumerate(tokens):
        if re.match(r"^\d+([\/\-]\d+)?[а-я]?$", t):
            return t, " ".join(tokens[:i] + tokens[i + 1:]).strip()
    return None, s


def extract_letter(s: str):
    tokens = s.split()
    for i in range(len(tokens) - 1):
        if any(re.match(p, tokens[i]) for p in LETTER_WORDS):
            return tokens[i + 1], " ".join(tokens[:i] + tokens[i + 2:]).strip()
    for i in range(1, len(tokens)):
        if re.match(r"^\d+$", tokens[i - 1]) and re.match(r"^[а-я]$", tokens[i]):
            return tokens[i], " ".join(tokens[:i] + tokens[i + 1:]).strip()
    return None, s


def build_addr(city, st_type, st_name, number=None, housing=None, structure=None, letter=None):
    parts = []
    if city:                parts.append(f"г {city}")
    if st_type and st_name: parts.append(f"{st_type} {st_name}")
    if number and number != "0":
        h = str(number)
        if housing and housing != "0": h += f" корп{housing}"
        if structure and structure != "0": h += f" стр{structure}"
        if letter and letter != "0": h += f" лит{letter}"
        parts.append(h)
    return " ".join(parts) if parts else np.nan


def parse_address_a(raw):
    s = clean_text(raw)
    if not s:
        return np.nan

    city, s = normalize_city(s)
    st_type, st_name, s = extract_street(s)
    house, s = extract_by_words(s, HOUSE_WORDS)
    building, s = extract_by_words(s, BUILDING_WORDS)
    structure, s = extract_by_words(s, STRUCT_WORDS)
    letter, s = extract_letter(s)

    base_house, house_letter = None, None
    if house:
        m = re.match(r"^(\d+)([а-я])?$", str(house))
        if m:
            base_house, house_letter = m.group(1), m.group(2)
        else:
            base_house = str(house)

    return build_addr(city, st_type, st_name,
                      number=base_house,
                      housing=building,
                      structure=structure,
                      letter=letter or house_letter)


def normalize_addresses_a(input_path: str, output_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df["addr_norm"] = df["gkh_address"].apply(parse_address_a)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "output/cleaned_source_a.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/cleaned_a.csv"
    normalize_addresses_a(src, out)
