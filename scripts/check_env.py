#!/usr/bin/env python3
"""P2: check GPU / package readiness before train."""

from __future__ import annotations


def main() -> None:
    print("=== SlaTriage env check ===")
    try:
        import torch

        print(f"torch={torch.__version__}")
        cuda = torch.cuda.is_available()
        print(f"cuda_available={cuda}")
        if cuda:
            print(f"device={torch.cuda.get_device_name(0)}")
            print(f"vram_gb≈{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
    except ImportError:
        print("torch: NOT INSTALLED")

    for pkg in ("transformers", "peft", "datasets", "yaml", "slither"):
        try:
            if pkg == "yaml":
                import yaml  # noqa: F401

                print("pyyaml: OK")
            elif pkg == "slither":
                import importlib.util

                print(
                    "slither: OK"
                    if importlib.util.find_spec("slither")
                    else "slither: NOT INSTALLED (P1: pip install slither-analyzer)"
                )
            else:
                m = __import__(pkg)
                print(f"{pkg}: OK ({getattr(m, '__version__', '?')})")
        except ImportError:
            print(f"{pkg}: NOT INSTALLED")

    print("=== done ===")
    print("Next: python scripts/smoke_test.py")


if __name__ == "__main__":
    main()
