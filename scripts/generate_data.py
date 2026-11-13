"""Generate the synthetic patient dataset (fixed seed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_genomic.data import generate_patients

OUT = Path(__file__).resolve().parents[1] / "data" / "patients.csv"


def main() -> None:
    df = generate_patients(n=5000, seed=42)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(df)} cols={len(df.columns)}")
    print(df["risk_5yr"].value_counts().to_dict())
    print(df["drug_responder"].value_counts().to_dict())


if __name__ == "__main__":
    main()
