"""End-to-end pipeline: raw CRF -> SDTM -> ADaM -> TLF tables."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import generate_raw
import generate_sdtm
import generate_adam
import generate_tlfs


def main() -> None:
    print("=" * 60)
    print("PyTLF pipeline: ABC-001")
    print("=" * 60)
    print("[1/4] Generating raw CRF data...")
    generate_raw.main()
    print("[2/4] Building SDTM domains...")
    generate_sdtm.main()
    print("[3/4] Building ADaM datasets...")
    generate_adam.main()
    print("[4/4] Generating TLF tables...")
    generate_tlfs.main()
    print("Done.")


if __name__ == "__main__":
    main()
