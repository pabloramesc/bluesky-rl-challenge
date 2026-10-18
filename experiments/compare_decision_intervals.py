"""Compare single-agent PPO decisions every 5, 10, and 20 seconds."""

from bluesky_pabloramesc.cli import run_comparison_cli

if __name__ == "__main__":
    run_comparison_cli(frequency=True)
