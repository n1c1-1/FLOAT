"""The names Inspect's service module (services/inspect.py) gave before it was split into sibling modules: every one still importable from
astro_gpu.client.services.inspect."""
import importlib

#: dir() of the module before the split (dunder names left out)
NAMES = (
    "Any", "ApiWorker", "CAMERA_PREVIEWS", "CATALOGUE_G_MAX", "CATALOGUE_MAX", "CalibrationData",
    "CalibrationReport", "ColorBundle", "FrameItem", "GRID_LINES", "GridLine", "GuiDiskWatch", "ImageCache",
    "InspectService", "MASTERS", "OrderedDict", "Path", "PixelSource", "ProjectFrames", "QCoreApplication",
    "QObject", "QTimer", "RegistrationBundle", "SKY_BLOCK", "ScaledWcs", "SessionRow", "ShownImage", "Signal",
    "SkyData", "StackBundle", "VisualSpec", "WeakKeyDictionary", "_DEC_STEPS", "_MISSING", "_RA_STEPS",
    "_WATCHED", "_array", "_calib_masters", "_calib_of", "_calibrated_colour", "_debayered", "_files",
    "_first_readable", "_hook", "_installed", "_keys_of", "_levels", "_lut", "_master_stats", "_masters",
    "_masters_lock", "_norm", "_offsets_anyway", "_one_plane", "_pieces", "_rgb_of_file", "_services",
    "_sky_model", "_stack_meta", "_stack_zone", "_step", "_stk", "_try", "_unavailable", "_w", "_watch",
    "_wcs_of", "_which_done", "add_sky", "annotations", "api", "apply_matrix", "array_source", "block_of",
    "browse_group", "calframe_source", "calib_checks", "calibrated_frame", "calibrated_source",
    "calibration_data", "calibration_left_out", "calibration_report", "catalogue_stars", "color_bundle",
    "color_fit_of", "color_source", "crop_frame", "dark_lines", "dataclass", "dec_label", "diag", "field",
    "final_sky_bundle", "final_wcs_source", "flat_pct", "forget_masters", "frame_stats", "frame_table_any",
    "gradient_source", "grey_sky", "histogram8", "inventory_headers", "light_thumb", "lights_of", "load_visual",
    "missing_of", "missing_text", "mtf", "neutral_params", "neutral_preview", "norm_path", "not_computed",
    "not_yet", "np", "os", "post_annotations", "post_key", "post_source", "preview_source", "preview_wcs_source",
    "project_frames", "ra_label", "re", "reduce2", "reference_frame_place", "registration_bundle",
    "registration_from_table", "removed_source", "render8", "render_place", "resolve_key", "sensor_size",
    "service_of", "set_calibration_excluded", "shown_neutral", "sky_bundle", "sky_data", "sky_grid",
    "stack_bundle", "stack_place", "stretch_image", "stretch_params", "sys", "threading", "time",
    "warm_annotations", "wcs_source",
)


def test_every_name_still_importable():
    m = importlib.import_module("astro_gpu.client.services.inspect")
    missing = [n for n in NAMES if not hasattr(m, n)]
    assert not missing, missing
