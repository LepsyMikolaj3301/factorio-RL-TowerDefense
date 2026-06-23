"""Deprecated shim — Tower Defense training moved into the `fle.rl` package.

Use the canonical entry point instead:

    python -m fle.rl.train --num-envs 4 --total-timesteps 1000000 --evolution-factor 0.5

This file is kept only for backward compatibility and forwards to fle.rl.train.
"""

from fle.rl.train import main

if __name__ == "__main__":
    main()
