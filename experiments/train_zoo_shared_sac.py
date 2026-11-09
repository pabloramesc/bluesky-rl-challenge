"""Train, resume, evaluate, or replay additive-attention SHARED_SAC."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("shared_sac", multi_agent=True)
