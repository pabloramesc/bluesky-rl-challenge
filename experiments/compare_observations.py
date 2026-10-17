"""Compare existing geometry in upstream, ego Cartesian, and goal frames."""

from bluesky_pabloramesc.cli import run_comparison_cli

if __name__ == "__main__":
    run_comparison_cli(observation=True)
