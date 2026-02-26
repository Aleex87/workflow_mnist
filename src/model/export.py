"""
This file exports the trained model to TorchScript format.

Responsibilities:
- Load trained model weights
- Convert to TorchScript (input: artifacts/digits_mlp_state_dict.pt)
- Save exported model inside artifacts (output: artifacts/digits_mlp.ts)
"""

from pathlib import Path
import torch

from model import DigitsMLP


def main() -> None:
    artifacts_dir = Path("artifacts")
    weights_path = artifacts_dir / "digits_mlp_state_dict.pt"
    out_path = artifacts_dir / "digits_mlp.ts"

    # Check that weights exist
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Missing weights at {weights_path}. Run: uv run python src/model/train.py"
        )

    # Recreate the model architecture
    model = DigitsMLP()

    # Load trained weights into the model
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Set evaluation mode
    model.eval()

    # TorchScript export (trace) using an example input
    example_input = torch.zeros(1, 64, dtype=torch.float32)
    traced = torch.jit.trace(model, example_input)

    # Save the TorchScript model
    traced.save(str(out_path))

    # Sanity check: load it back and run one forward pass
    loaded = torch.jit.load(str(out_path), map_location="cpu")
    _ = loaded(example_input)

    print(f"Export completed. TorchScript saved to: {out_path}")


if __name__ == "__main__":
    main()