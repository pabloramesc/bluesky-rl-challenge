"""Train context-token attention once and compare with read-only reference runs."""

from .architecture_tree import attention_comparison_tree
from .reference_comparison import run_reference_comparison


def context_attention_tree():
    """Keep the original reference names and add one context-token variant."""
    tree = attention_comparison_tree()
    tree["children"].append(
        {
            "name": "context_transformer",
            "settings": {
                "architecture": "context_entity_transformer",
                "pooling": "context_token",
                "transformer_layers": 1,
            },
            "children": [{"name": "none", "settings": {"temporal": "none"}}],
        }
    )
    return tree


def run_context_attention_comparison(args):
    """Reuse the original three attention variants and train only context attention."""
    return run_reference_comparison(
        args,
        context_attention_tree()["children"][-1],
        reference_tree=attention_comparison_tree(),
        label="Context attention",
    )
