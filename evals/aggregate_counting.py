"""Back-compat shim: moved to evals.contrastive.aggregate_counting (2026-08-19 reorg)."""
from .contrastive.aggregate_counting import *  # noqa: F401,F403
from .contrastive.aggregate_counting import main  # noqa: F401

if __name__ == "__main__":
    main()
