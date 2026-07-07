from _common import load_args
from ioh.pipeline import build_vitaldb_manifest


if __name__ == "__main__":
    build_vitaldb_manifest(load_args())

