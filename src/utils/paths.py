
import os

CWD = os.path.dirname(__file__)
PATH_DATA = os.path.join(CWD, "../../data")
PATH_RECORDS = os.path.join(PATH_DATA, "records/")
PATH_STATIC = os.path.join(PATH_DATA, "static/")


def icon_path(name: str):
    return os.path.join(PATH_STATIC, f"{name}.png")


# TODO 1: remove custom data format - use zipped pickles or something sane
# TODO 2: mode to recorder module
def record_path(name: str):
    return os.path.join(PATH_RECORDS, f"{name}.tsr")

def last_record_path(pos:int = 0):
    records = os.listdir(PATH_RECORDS)
    records.sort(reverse=True)
    if records:
        return os.path.join(PATH_RECORDS, records[pos])
    return None