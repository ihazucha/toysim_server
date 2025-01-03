from utils.data import SimData, last_record_path
from modules.recorder import RecordReader

import cv2
from time import sleep

T = 1.0 / 30.0

if __name__ == "__main__":
    record_path = last_record_path()
    if record_path is None:
        print("[Planner] No records found")
        exit()
    data: list[SimData] = RecordReader.read(record_path=record_path)
    for d in data:
        rgb_image = cv2.cvtColor(d.camera_data.rgb_image, cv2.COLOR_BGR2RGB)
        cv2.imshow("RGB Image", rgb_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        sleep(T)

    cv2.destroyAllWindows()