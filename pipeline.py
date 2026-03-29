"""
pipeline.py — МТС True Tech Cup 2026 «Выше крыши»

python pipeline.py src_A.csv src_B.csv [output_dir]
"""

import subprocess
import sys
from pathlib import Path

from clean_source_a import clean_source_a
from clean_source_b import clean_source_b
from normalize_addr_a import normalize_addresses_a
from normalize_addr_b import normalize_addresses_b
from matching import build_final


def run_pipeline(src_a: str, src_b: str, outdir: str = "./output"):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("1/5 Очистка A...")
    clean_source_a(src_a, output_path=str(out / "cleaned_source_a.csv"))

    print("2/5 Очистка B...")
    clean_source_b(src_b, output_path=str(out / "cleaned_source_b.csv"))

    print("3/5 Адреса A...")
    normalize_addresses_a(str(out / "cleaned_source_a.csv"), output_path=str(out / "cleaned_a.csv"))

    print("4/5 Адреса B...")
    normalize_addresses_b(str(out / "cleaned_source_b.csv"), output_path=str(out / "cleaned_b.csv"))

    print("5/6 Матчинг и сборка признаков...")
    join_file = out / "final_buildings_with_addr_and_new_features.csv"
    build_final(
        path_a=str(out / "cleaned_a.csv"),
        path_b=str(out / "cleaned_b.csv"),
        output_path=str(join_file),
    )

    print("6/6 Обучение RF...")
    script = Path(__file__).parent / "rf_train.py"
    subprocess.run(
        [sys.executable, str(script), str(join_file)],
        check=True,
    )

    print(f"\nГотово! Результаты в {out.resolve()}")


if __name__ == "__main__":
    src_a  = sys.argv[1] if len(sys.argv) > 1 else "input/cup_it_example_src_A.csv"
    src_b  = sys.argv[2] if len(sys.argv) > 2 else "input/cup_it_example_src_B.csv"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "./output"
    run_pipeline(src_a, src_b, outdir)