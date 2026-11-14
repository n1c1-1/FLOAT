"""The new graph page alone, for development: ``python -m astro_gpu.client.graphview [PROJECT]``.

PROJECT is a FLOAT project folder or its project file; its definition is converted to a graph and shown. Nothing is
ever written: the page edits a copy held in memory (with its own undo), the project stays as it is. Without PROJECT, a
blank page. ``--fr`` shows the French texts."""
from __future__ import annotations

import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    from PySide6.QtWidgets import QApplication

    from .. import i18n
    from ..icons import app_icon
    from ..widgets import QSS
    from .document import MemoryDocument, open_project
    from .page import GraphPage
    lang = "fr" if "--fr" in argv else "en"
    argv = [a for a in argv if a != "--fr"]
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyleSheet(QSS)
    app.setWindowIcon(app_icon())
    i18n.set_language(lang)
    document = open_project(argv[0]) if argv else MemoryDocument()
    page = GraphPage(document)
    page.setWindowTitle("FLOAT · " + i18n._("Graph page") + " · " + i18n._("read only"))
    page.resize(1600, 900)
    page.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
