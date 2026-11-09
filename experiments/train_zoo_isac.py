"""Train, resume, evaluate, or replay additive-attention ISAC."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("isac", multi_agent=True)
