"""The names operations gave before its gestures were split by family into modules: every one still importable from
astro_gpu.graphcore.operations, the registry of gestures holding the same names, each the function of the module that
now holds it."""
import importlib

#: dir() of operations before the split (dunder names left out)
NAMES = (
    "ALIASES", "COMMON", "FILE_KINDS", "FRAME_KINDS", "Index", "JOIN_KINDS", "LABEL", "LAYER_PORTS", "M",
    "NO_SELECTION", "OFFABLE", "OPS", "Out", "PORT_CHAINS", "POST_PORT", "Refusal", "Result", "Row",
    "SELECTION_KEYS", "SYNTH_IN", "SYNTH_PORT", "TR", "Tap", "VIEW_OPS", "WHOLE_VARIANT", "Wire", "_acted",
    "_add_calibration", "_add_join", "_add_port_row", "_after_subtree", "_alloc", "_apply_change", "_apply_many",
    "_auto_align", "_blk", "_branch", "_calib_block", "_calibrated_as_one", "_carry_excluded", "_channel",
    "_clone_block", "_copy_block", "_copy_calibration", "_copy_flow", "_created", "_dedup", "_drawn", "_eff_list",
    "_engine_name", "_entry", "_exclude_in", "_file_name", "_files", "_first", "_flow_chain", "_flow_ends",
    "_flow_of", "_flow_sig", "_flows_through", "_follow_flow_pauses", "_follow_stack", "_forget_night",
    "_holder_of", "_ids", "_image_name", "_input", "_insert", "_insert_after_frames", "_is_layers_row",
    "_join_problem", "_kind", "_legacy_chains", "_list_of", "_make_variant", "_map_port", "_mid_row", "_model",
    "_named", "_names", "_next_sid", "_night", "_night_off", "_night_row", "_no_clash", "_norm_align",
    "_norm_names", "_norm_params", "_norm_taps", "_not_into_whole_variant", "_not_shared", "_port_row",
    "_post_tap", "_put", "_recipe_said", "_reimage", "_remove_calibration", "_row", "_run", "_session",
    "_sessions_state", "_set_list", "_shared", "_stack_calibrated", "_stage_names", "_stages_of", "_store",
    "_store_all", "_styled", "_summarise", "_sync_shared", "_synth_on", "_thaw", "_to_post", "_unhide",
    "_variant_parts", "_view_op", "_whole_variants_follow", "apply", "apply_many", "attempt", "check",
    "check_param", "clone", "combos", "copy", "dataclass", "delete_rows", "dependants", "detach", "edges",
    "evaluate", "excluded_files", "finish", "fold_guard", "fold_problem", "free_name", "fusion_legacy",
    "in_ports", "inherited", "join", "json_of", "layers_free", "legality", "model_json", "name_of", "natural",
    "need_greyed", "new_align", "new_block", "new_id", "new_view", "normalise", "op_add_block", "op_add_branch",
    "op_add_combination", "op_add_files", "op_add_night", "op_add_row", "op_add_stack",
    "op_add_stars_combination", "op_add_variant", "op_apply_to_rows", "op_delete_row", "op_discard_variant",
    "op_duplicate_row", "op_exclude_files", "op_exclude_frames", "op_follow_session", "op_keep_variant",
    "op_merge_nights", "op_merge_rows", "op_move_block", "op_move_files", "op_move_night", "op_new_night",
    "op_own_grid", "op_remove_block", "op_remove_files", "op_remove_night", "op_remove_own_calibration",
    "op_remove_session", "op_remove_tap", "op_remove_with_dependants", "op_rename_night", "op_rename_row",
    "op_rename_session", "op_rename_tap", "op_replace_chain", "op_reset_own", "op_reset_own_files",
    "op_send_to_post", "op_set_align", "op_set_align_off", "op_set_calibrate_together", "op_set_combination_type",
    "op_set_common_files", "op_set_common_setting", "op_set_flow_options", "op_set_flow_param",
    "op_set_flow_pauses", "op_set_image_file", "op_set_input", "op_set_left_out", "op_set_method",
    "op_set_night_filter", "op_set_own_calibration", "op_set_param", "op_set_pause", "op_set_recipe_value",
    "op_set_reference", "op_set_row_param", "op_set_session_files", "op_set_session_filter", "op_set_setting",
    "op_share_night", "op_split_by", "op_split_night", "op_store_params", "op_switch_off", "op_take_alternative",
    "op_try_variant", "op_unwire", "op_update_recipe", "op_use_for_all_nights", "op_wire", "out_ports",
    "own_spot", "owned_stages", "port_chains", "rank", "replace", "rules", "say_text", "session_name",
    "title_msg", "try_op", "typed", "wfl", "whole_original", "would_cycle",
)

#: the gestures of the registry OPS before the split
GESTURES = (
    "add_row", "delete_row", "add_night", "remove_night", "new_night", "add_stack", "share_night", "move_night",
    "merge_nights", "split_night", "merge_rows", "duplicate_row", "rename_row", "rename_night", "set_reference",
    "set_align", "own_grid", "set_align_off", "set_calibrate_together", "add_block", "add_variant", "move_block",
    "remove_block", "switch_off", "set_pause", "set_setting", "apply_to_rows", "set_combination_type", "wire",
    "unwire", "take_alternative", "send_to_post", "remove_tap", "add_files", "remove_files", "use_for_all_nights",
    "set_common_files", "set_common_setting", "reset_own", "reset_own_files", "move_files", "split_by",
    "set_param", "set_row_param", "set_method", "set_own_calibration", "follow_session", "remove_own_calibration",
    "set_night_filter", "add_combination", "update_recipe", "set_recipe_value", "set_input",
    "add_stars_combination", "try_variant", "keep_variant", "discard_variant", "set_session_files",
    "rename_session", "set_session_filter", "set_left_out", "remove_session", "exclude_frames", "exclude_files",
    "set_image_file", "rename_tap", "set_flow_pauses", "set_flow_options", "set_flow_param", "store_params",
    "replace_chain", "add_branch", "remove_with_dependants",
)


def test_every_name_still_importable():
    ops = importlib.import_module("astro_gpu.graphcore.operations")
    missing = [n for n in NAMES if not hasattr(ops, n)]
    assert not missing, missing


def test_the_registry_holds_the_same_gestures():
    from astro_gpu.graphcore import operations as ops
    assert sorted(ops.OPS) == sorted(GESTURES)
    for k, fn in ops.OPS.items():
        assert getattr(ops, "op_" + k) is fn
        assert getattr(importlib.import_module(fn.__module__), "op_" + k) is fn
