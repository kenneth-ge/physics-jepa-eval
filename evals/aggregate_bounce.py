"""Back-compat shim: moved to evals.contrastive.aggregate_bounce (2026-08-19 reorg)."""
from .contrastive.aggregate_bounce import *  # noqa: F401,F403
from .contrastive.aggregate_bounce import main  # noqa: F401

if __name__ == "__main__":
    main()
