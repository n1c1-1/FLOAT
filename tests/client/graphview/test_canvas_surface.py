"""The names the graph page's canvas module gave before its widgets moved to their own module: every one still
importable from astro_gpu.client.graphview.canvas, the moved widgets being the same classes."""
import importlib

#: dir() of canvas before the move (dunder names left out)
NAMES = (
    "ACCENT", "ALERT_MS", "BH", "BandItem", "BlockItem", "CHAIN_STYLES", "CLIENT_KEY", "CaptionItem", "DIM",
    "EDGE", "EDGE_WAIT", "EMPTY_COL", "FALLBACK_WORD", "FILE_KINDS", "FILE_LABEL", "FileTarget", "GhostItem",
    "GraphCanvas", "IMAGE_COL", "Index", "JOIN_KINDS", "LINK_MS", "LinkItem", "MARGIN", "MAX_AUTO", "MAX_ZOOM",
    "MIME_KIND", "MIN_AUTO", "MIN_ZOOM", "MODEL_SUMMARY", "Minimap", "NO_OUT", "NameEdit", "OPTIONAL_PORTS",
    "OUT_COL", "PORT_COL", "PROJECT", "ProjectItem", "QApplication", "QColor", "QEvent", "QFont", "QFrame",
    "QGraphicsScene", "QGraphicsView", "QKeyEvent", "QLabel", "QLineEdit", "QMenu", "QPainter", "QPainterPath",
    "QPen", "QPoint", "QPointF", "QPushButton", "QRect", "QRectF", "QTimer", "QTransform", "QVBoxLayout",
    "QWidget", "Qt", "Result", "RubberWire", "SIDE", "SIDE_IN", "SIDE_INPUTS", "SIDE_KINDS", "STAR_OUTPUTS",
    "STATE_COL", "SYNTH_PORT", "Signal", "Spinner", "StackItem", "TipItem", "Toast", "VIEW_KEY", "WHEEL_STEP",
    "WIRE_WORD", "WireItem", "ZOOMS", "_", "_KIND", "_fields", "annotations", "block_menu", "html", "kind_what",
    "number", "pinch_factor", "plural", "prefs", "shown", "state_word", "style", "textwrap", "time", "title_msg",
    "touched_blocks", "wheel_gesture",
)


def test_every_name_still_importable():
    cv = importlib.import_module("astro_gpu.client.graphview.canvas")
    missing = [n for n in NAMES if not hasattr(cv, n)]
    assert not missing, missing


def test_the_moved_widgets_are_the_same_classes():
    from astro_gpu.client.graphview import canvas, canvas_overlays
    for n in ("Toast", "Minimap", "NameEdit", "LINK_MS"):
        assert getattr(canvas, n) is getattr(canvas_overlays, n)
