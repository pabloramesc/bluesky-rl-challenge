# Upstream dependency

Repository: https://github.com/TUDelft-CNS-ATM/bluesky-gym

Pinned baseline: `00930013219af4c17e509c3efc84a8becd0f3546` (original branch: `AI4REAL-NET-Competition`).

The original repository and Git history are preserved in the submodule at
`vendor/bluesky-gym`. The development repository has its own Git history.
Upstream files and LICENSE are unchanged. Personal code lives in `experiments/`.

## Restore the pinned version

```bash
git submodule update --init --recursive
```

## Update deliberately

Fetch upstream, select a reviewed commit, and record the new pointer:

```bash
git -C vendor/bluesky-gym fetch origin
git -C vendor/bluesky-gym checkout <COMMIT>
git add vendor/bluesky-gym
git commit -m "Update BlueSky-Gym dependency"
```

Reinstall the editable dependency if its dependencies or packaging changed.
Keep personal wrappers and agents outside the submodule. If you need environment
changes, commit them in a fork of the submodule, push those commits, and change
`.gitmodules` to that fork so other clones can retrieve them.

## Migration backup

The original Git metadata was copied outside this workspace to
`/home/pablo/workspace/bluesky-gym-git-backup-20261017-200243` before migration. Existing virtual environments and generated
artifacts were preserved. Historical result logs may retain old paths.
