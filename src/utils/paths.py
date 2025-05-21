from pathlib import Path

_CWD = Path(__file__).parent
PATH_STORAGE = _CWD.parents[1] / "storage"
PATH_RECORDS = PATH_STORAGE / "records"
PATH_STATIC = PATH_STORAGE / "static"

# TODO 1: remove custom data format - use zipped pickles or something sane
# TODO 2: move to recorder module
def record_path(name: str):
    return str(PATH_RECORDS / f"{name}.pickle")

def last_record_path(pos: int = 0):
    records = sorted(PATH_RECORDS.iterdir(), reverse=True)
    if records:
        return records[pos]
    return None