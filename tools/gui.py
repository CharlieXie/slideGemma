#!/usr/bin/env python3
"""Launch the slideGemma desktop GUI.

Usage::

    python tools/gui.py
"""

from __future__ import annotations

import sys


def main():
    from PySide6.QtWidgets import QApplication
    from slide_gemma.gui.app import Launcher

    app = QApplication(sys.argv)
    window = Launcher()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
