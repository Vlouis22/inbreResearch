"""
Standalone entry point for the Lab Agent.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline.lab_pipeline import LabInput, load_lab_models, run_lab_pipeline
from schemas.lab_profile import LabProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


EXAMPLE_LAB_INPUTS: list[LabInput] = [
    LabInput(
        source_type="digital_table",
        content=(
            "CMP\n"
            "Test,Result,Unit,Reference Range,Flag\n"
            "Sodium,137,mmol/L,135-145,N\n"
            "Potassium,3.1,mmol/L,3.5-5.1,L\n"
            "Creatinine,1.7,mg/dL,0.6-1.3,H\n"
            "AST,68,U/L,10-40,H\n"
            "ALT,74,U/L,7-56,H\n"
            "Albumin,3.1,g/dL,3.5-5.0,L\n"
        ),
    )
]


def main() -> None:
    print("  Lab Agent — Structured Lab Profile Generator")
    print("\n")

    load_lab_models()

    logger.info("Processing %d lab input(s)...", len(EXAMPLE_LAB_INPUTS))
    profile: LabProfile = run_lab_pipeline(EXAMPLE_LAB_INPUTS)

    print("\n")
    print("  LAB PROFILE")
    print("\n")
    print(profile.model_dump_json(indent=2))
    print("\n")


if __name__ == "__main__":
    main()
