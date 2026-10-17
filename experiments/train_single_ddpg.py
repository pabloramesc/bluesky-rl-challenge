"""Train or replay the single-agent DDPG MLP baseline."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("ddpg", multi_agent=False)
