import zmq
from importlib import reload
import sys
import signal
import os

def main_loop():
    """
    Executing this in loop can help during development
    by shortening time of repeated shutting down and opening
    of the app - cosmetic changes of Qt elements can be seen
    after the shortened restart. For bigger changes and adding
    new elements, proper shutdown is still required
    """
    while True:
        main()


def main():
    # Force reload
    if "modules.render" in sys.modules:
        import modules.render

        reload(modules.render)

    from modules.render import Renderer

    try:
        renderer = Renderer()
        renderer.run()
    finally:
        zmq.Context.instance().destroy()
        print("Exiting")


if __name__ == "__main__":
    main()
