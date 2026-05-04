"""Compatibility CLI for evaluation runs.

The implementation is split behind this stable module so existing commands such
as `python -m eval.run_eval` and tests importing private helpers keep working.
"""

from . import _runtime as _runtime

for _name, _value in vars(_runtime).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

main = _runtime.main


if __name__ == "__main__":
    main()
