"""Train or replay the single-agent SAC MLP baseline."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("sac", multi_agent=False)
