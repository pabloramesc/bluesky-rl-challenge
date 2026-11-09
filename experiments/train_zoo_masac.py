"""Train, resume, evaluate, or replay additive-attention MASAC."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("masac", multi_agent=True)
