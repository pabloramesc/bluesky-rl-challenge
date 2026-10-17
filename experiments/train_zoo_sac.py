"""Train or replay the multi-agent SAC MLP baseline."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("sac", multi_agent=True)
