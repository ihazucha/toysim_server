# import libraries
from vidgear.gears.asyncio import NetGear_Async
import cv2, asyncio
import time
import numpy as np

client = NetGear_Async(
    protocol="tcp",
    address="192.168.0.105",
    port="5454",
    pattern=2,
    receive_mode=True,
    # ---
    # logging=True,
).launch()


async def main():
    start = time.time()
    count = 0
    data_size = None
    try:
        async for frame in client.recv_generator():
            if data_size is None and frame is not None:
                data_size = frame.size
            count += 1
            cv2.imshow("Output Frame", cv2.flip(frame, 0))
            key = cv2.waitKey(1) & 0xFF
            await asyncio.sleep(0)
    finally:
        dt = time.time() - start
        Bps = ((data_size * count) / dt)
        MBps = Bps / 1_000_000
        print(
            f"Fsize  :{data_size:12} [B]\n"
            f"Fcount :{count:12}\n"
            f"Dur    :{dt:12.2f} [s]\n"
            f"Speed  :{MBps:12.2f} [MBps] | ({(MBps * 8):.0f} [Mbps])"
        )

if __name__ == "__main__":
    asyncio.set_event_loop(client.loop)
    try:
        client.loop.run_until_complete(main())
    except (KeyboardInterrupt, SystemExit):
        pass

    cv2.destroyAllWindows()
    client.close()