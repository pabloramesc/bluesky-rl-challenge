"""Train or replay the multi-agent PPO MLP baseline."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("ppo", multi_agent=True)
