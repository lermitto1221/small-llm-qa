"""Run a synthetic scoring example without a GPU or external dependencies."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from small_llm_qa.metrics import score_records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional path for synthetic scores")
    args = parser.parse_args()
    refs = json.loads((ROOT / "examples" / "demo_references.json").read_text(encoding="utf-8"))
    predictions = json.loads((ROOT / "examples" / "demo_predictions.json").read_text(encoding="utf-8"))
    report = score_records(refs, predictions)
    if report["metrics"]["total_count"] != 6:
        raise RuntimeError("The synthetic fixture is incomplete")
    print("Synthetic scoring demo: fixed example answers; no model inference.")
    print(json.dumps(report["metrics"], indent=2, allow_nan=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
