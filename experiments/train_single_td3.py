"""Train or replay the single-agent TD3 MLP baseline."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("td3", multi_agent=False)
