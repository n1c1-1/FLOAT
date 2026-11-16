"""The names the viewer's module (client/imageview.py) gave before it was split into sibling modules: every one still importable from
astro_gpu.client.imageview."""
import importlib

#: dir() of the module before the split (dunder names left out)
NAMES = (
    "ACCENT", "APP_NAME", "Alignment", "Any", "BAD", "CANVAS_BG", "CANVAS_RGB", "CHANNEL_COLORS", "Display",
    "ElidedLabel", "FORMATS", "FilmStrip", "HistogramView", "IconLabel", "IconText", "ImageCanvas",
    "ImageExportDialog", "ImageView", "InfoBand", "Loader", "MONO_COLOR", "MONO_FONT", "OK", "OrderedDict",
    "Path", "Placement", "Prepared", "QApplication", "QButtonGroup", "QCheckBox", "QColor", "QComboBox",
    "QDialog", "QDoubleSpinBox", "QEvent", "QFileDialog", "QFont", "QFontMetricsF", "QFrame", "QGridLayout",
    "QGuiApplication", "QHBoxLayout", "QImage", "QKeySequence", "QLabel", "QLineEdit", "QMenu", "QObject",
    "QPainter", "QPainterPath", "QPen", "QPointF", "QPolygonF", "QPushButton", "QRadioButton", "QRectF", "QSS",
    "QScrollArea", "QScrollBar", "QShortcut", "QSizePolicy", "QSpinBox", "QStandardPaths", "QTILES", "QTimer",
    "QVBoxLayout", "QWidget", "Qt", "RAW_SUFFIXES", "RTILES", "SKY_GRID", "SideScroll", "Signal", "StretchPanel",
    "SyncGroup", "TAG_BG", "TILE", "Unavailable", "ViewImage", "WARN", "ZOOM_PRESETS", "ZOOM_STEP", "_", "_Lru",
    "_UID", "_default_loader", "_identity", "_mirror_state", "_place_of", "_read", "_sep", "_sky_map",
    "_state_of", "_to_side", "_wcs_of", "_write_fits", "alignment", "annotations", "ascii_text", "cap_dialog",
    "dataclass", "default_loader", "default_stretch", "draw", "export_image", "exposure_text", "field",
    "format_of", "heapq", "icon_only", "itertools", "kind_of", "label", "language", "log_detail", "math",
    "memory_budget", "menu_button", "nice_ticks", "np", "number", "os", "pinch_factor", "plain_error", "planar",
    "prepare", "quick_thumbnail", "re", "read_file", "set_icon", "short_number", "sky_text", "slug", "sm",
    "spaced_ticks", "tag", "threading", "thumbnail_of", "time", "time_text", "to_qimage", "value_text",
    "wheel_gesture",
)


def test_every_name_still_importable():
    m = importlib.import_module("astro_gpu.client.imageview")
    missing = [n for n in NAMES if not hasattr(m, n)]
    assert not missing, missing
