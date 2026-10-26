"""Train, resume, evaluate, or replay separate aircraft PPO policies."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("ippo", multi_agent=True)
