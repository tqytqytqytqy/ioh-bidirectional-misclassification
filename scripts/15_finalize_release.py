from _common import load_args
from ioh.pipeline import finalize_release


if __name__ == "__main__":
    finalize_release(load_args())
