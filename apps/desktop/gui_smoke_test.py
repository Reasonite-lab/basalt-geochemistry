"""Open the real Tk application, run the bundled example, and close it."""
from __future__ import annotations

import importlib.machinery
import importlib.util
import time
from pathlib import Path


def load_gui_module():
    path = Path(__file__).resolve().parent / "d36_predictor_gui.pyw"
    loader = importlib.machinery.SourceFileLoader("d36_predictor_gui", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    if spec is None:
        raise RuntimeError("Could not create GUI module spec")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def wait_until(app, predicate, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.update()
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError(f"GUI smoke timeout while waiting for {label}")


def main() -> None:
    module = load_gui_module()
    app = module.D36App()
    app.withdraw()
    try:
        wait_until(app, lambda: app.engine is not None, 30.0, "engine")
        app._load_example()
        app._start_analysis()
        wait_until(app, lambda: app.result is not None, 60.0, "analysis")
        assert len(app.probability_tree.get_children()) == 3
        assert len(app.region_tree.get_children()) == 234
        assert len(app.reference_tree.get_children()) == 1689
        assert app.last_output is not None and app.last_output.exists()
        print(f"GUI_SMOKE_PASS output={app.last_output}")
    finally:
        app.destroy()


if __name__ == "__main__":
    main()

