"""
Command-line interface for exporting trained models to deployment formats.

This script wraps the export utilities and provides a simple CLI for generating
TorchScript (.pt) and ONNX (.onnx) artifacts from a trained model checkpoint.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running this script directly (python scripts/export_model.py) by
# ensuring the project root is on sys.path so `src.*` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.export import export_torchscript, export_onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Parse CLI arguments and export the model to TorchScript and ONNX formats.

    Raises:
        FileNotFoundError: If the checkpoint path does not exist.
    """
    
    parser = argparse.ArgumentParser(description="Export model to deployment formats.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    out_dir.mkdir(parents=True, exist_ok=True)

    export_torchscript(checkpoint, out_dir / "model_torchscript.pt")
    export_onnx(checkpoint, out_dir / "model_onnx.onnx")

    logger.info(f"Export complete. Files saved to: {out_dir}")

if __name__ == "__main__":
    main()
