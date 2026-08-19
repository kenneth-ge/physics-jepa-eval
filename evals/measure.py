"""Back-compat shim: moved to evals.contrastive.measure (2026-08-19 reorg)."""
from .contrastive.measure import *  # noqa: F401,F403
from .contrastive.measure import main  # noqa: F401

if __name__ == "__main__":
    main()
