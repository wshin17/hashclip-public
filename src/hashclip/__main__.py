"""Allow ``python -m hashclip`` as a shortcut for ``python -m hashclip.cli``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
