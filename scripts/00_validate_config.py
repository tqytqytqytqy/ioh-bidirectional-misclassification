from _common import load_args
from ioh.pipeline import validate_config


if __name__ == "__main__":
    validate_config(load_args())

