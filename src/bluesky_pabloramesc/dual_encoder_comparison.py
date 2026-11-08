"""Compare the dual Transformer with completed architecture reference results."""

from .reference_comparison import run_reference_comparison


def run_dual_encoder_comparison(args):
    """Train two-layer traffic/obstacle encoders with the reference PPO protocol."""
    return run_reference_comparison(
        args,
        {
            "name": "dual_transformer",
            "settings": {
                "architecture": "dual_entity_transformer",
                "pooling": "context_token",
                "transformer_layers": 2,
                "temporal": "none",
            },
        },
        label="Dual encoder",
    )
