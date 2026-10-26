"""Train, resume, or evaluate custom MAPPO with a centralized MLP critic."""

from bluesky_pabloramesc.cli import run_cli

if __name__ == "__main__":
    run_cli("mappo", multi_agent=True)
