# The tests of today's graph page, after the swap

The new graph page replaced `client/graph.py` and `client/blocks.py`; eleven test files drove them and could not import
any more. Each of their tests is one line here:

- **COVERED**: a test of the new page or of the graph core already checks the same behaviour (named).
- **PORTED**: rewritten on the new page and the project model (named; `legacy.py` is their bench).
- **GONE**: a detail of the old drawing code that no longer exists (said which).
- **LOST**: something the user could do or see that the new page does not do: not tested here, listed with how to
  reproduce it in the list of defects of the swap (its number, `s6a-N`).

Files: `misc` = test_gv_legacy_misc.py, `page` = test_gv_legacy_page.py, `flows` = test_gv_legacy_flows.py,
`model` = test_gv_legacy_model.py (all in tests/client/graphview/).

## test_graph47_client.py (50)
| old test | verdict | where |
|---|---|---|
| test_the_palette_places_a_block_at_its_usual_place_and_twice | PORTED | page::test_the_palette_places_a_block_at_its_usual_place_and_twice |
| test_a_blank_page_starts_from_a_session_and_builds_block_by_block | COVERED | test_gv_page.py::test_s01_double_click_session_makes_a_stack (a Session is placed with its stack); tests/graphcore/test_engine_rules.py::test_needs_are_the_engines (Color needs Astrometry) |
| test_a_workflow_saved_as_a_template_comes_back_in_the_gallery_and_as_a_file | LOST | s6a-18 (a saved template is offered only on a blank page; no window, no template file) |
| test_a_filters_template_on_a_new_project_makes_its_empty_sessions | PORTED | page::test_a_filters_template_on_a_new_project_makes_its_empty_sessions |
| test_a_color_block_of_a_mono_workflow_is_greyed_with_its_reason | LOST | s6a-7 |
| test_what_the_play_test_found_stays_fixed | LOST | s6a-7 (no Color in mono); the refused delete keeping the selection: page::test_a_refused_delete_keeps_the_selection_and_frames_what_needs_it |
| test_a_template_takes_the_place_of_an_empty_workflow_and_channels_are_not_post_layers | PORTED | flows::test_filters_make_one_workflow_each_and_their_combinations (channels give Post no layer); the empty workflow made from sessions is GONE (a Session is a stack) |
| test_branches_parallel_flows_removable_wires_and_the_palette_taking_a_block_back | PORTED | page::test_a_wire_to_post_is_taken_away_with_del_and_stays_away, page::test_a_block_dragged_onto_the_palette_is_removed; branch and parallel stack COVERED by test_gv_page.py::test_s06_branch_strip_makes_a_variant_below, ::test_s03_drag_session_onto_plus_session |
| test_the_first_blocks_have_their_cross_and_the_last_session_can_go | LOST | s6a-19 (Calibration not deletable, owner's call); the last Session: page::test_the_last_session_goes_with_its_workflow_and_ctrl_z_brings_everything_back |
| test_a_stage_phase_moves_the_bar_from_its_first_file | PORTED | misc::test_a_stage_phase_moves_the_bar_from_its_first_file |
| test_the_second_play_test_stays_fixed | PORTED | page::test_a_wire_to_post_is_taken_away_with_del_and_stays_away (stays away, a chosen wire kept over a rebuild); Combine dragged: tests/graphcore/test_core.py::test_places_lit_match_apply; branch method: test_graphmodel_facades.py::test_a_new_branch_from_a_block_and_beside_a_night_calibration |
| test_a_fusion_channel_session_removed_keeps_the_project_valid_and_the_fusion_red | LOST | s6a-10 |
| test_a_parallel_flow_on_a_template_without_night_and_a_calibration_in_the_second_flow | LOST | s6a-5 (the pause on a session's Calibration); a stack always has its night now |
| test_a_block_placed_twice_has_its_own_settings_and_its_own_pause | PORTED | page::test_a_block_placed_twice_has_its_own_settings_and_its_own_pause |
| test_a_pause_stays_with_a_block_switched_off_and_works_on_a_sessions_calibration | LOST | s6a-5 (Calibration); the switched-off part: page::test_a_pause_stays_with_a_block_switched_off |
| test_the_answers_to_the_design_questions | PORTED | page::test_a_block_placed_again_starts_from_its_defaults_and_one_switched_off_is_not_placed_twice; (1) tests/graphcore/test_gestures3b.py::test_a_mid_row_block_to_post_is_a_workflow_of_its_own_and_goes_with_its_wire; (3.7) test_gv_page2.py::test_s21_del_on_a_needed_block_is_refused_with_its_reason |
| test_a_wire_taken_away_stays_away_and_what_is_sent_after_still_goes_to_post | PORTED | page::test_a_wire_to_post_is_taken_away_with_del_and_stays_away |
| test_branch_names_are_numbered_as_names_not_by_their_blocks_number | PORTED | flows::test_branch_names_are_numbered_as_names_not_by_their_blocks_number |
| test_a_blank_page_has_a_link_to_the_templates_in_place_of_the_graph | COVERED | test_gv_parity.py::test_the_blank_page_link_opens_the_gallery_and_a_template_makes_its_workflow |
| test_a_project_with_a_workflow_has_a_new_workflow_link_that_opens_its_window | LOST | s6a-18 |
| test_the_template_cards_take_the_height_of_their_content_in_a_grid_that_wraps | PORTED | page::test_the_template_cards_take_the_height_of_their_content_in_a_grid_that_wraps |
| test_the_filters_card_makes_the_palette_chosen_on_it | PORTED | page::test_the_filters_card_makes_the_palette_chosen_on_it |
| test_the_dual_band_x2_card_offers_no_combination_choice | PORTED | page::test_the_dual_band_x2_card_offers_no_combination_choice |
| test_no_template_asks_for_a_file_it_only_places_its_blocks | PORTED | page::test_no_template_asks_for_a_file_it_only_places_its_blocks (the window's part: s6a-18) |
| test_several_images_are_all_numbered_the_first_too | LOST | s6a-8 (the Image's line); the names: page::test_images_are_numbered_by_their_workflows |
| test_an_image_added_beside_one_numbers_both_as_their_blocks | PORTED | page::test_images_are_numbered_by_their_workflows (an Image block's title is its kind; its row's caption carries the name) |
| test_deleting_a_middle_image_keeps_each_blocks_title_its_workflows_name | PORTED | page::test_images_are_numbered_by_their_workflows |
| test_the_menu_of_another_workflows_block_stays_open | PORTED | page::test_the_menu_of_another_workflows_block_stays_open |
| test_post_saving_its_layers_keeps_the_workflows_not_computed_yet | LOST | s6a-11 (after a block's image sent to Post); with a branch: page::test_post_saving_its_layers_keeps_the_workflows_not_computed_yet |
| test_an_image_dropped_is_placed_empty_and_the_new_flow_zone_offers_no_template | PORTED | page::test_an_image_placed_waits_for_its_file_and_is_not_run_without_it |
| test_a_session_chosen_beside_an_image_gets_the_frame_blocks_and_sources_have_their_cross | PORTED | page::test_an_image_and_a_session_have_their_cross_and_ctrl_z_brings_either_back |
| test_a_block_in_the_middle_sends_its_image_to_post | COVERED | test_gv_page.py::test_s10_to_post_adds_one_wire_no_row; once sent, not again: page::test_a_port_pulled_again_says_its_image_is_a_layer_already_and_a_shared_block_is_named_by_its_owner |
| test_after_the_start_no_block_opens_the_template_window | COVERED | test_gv_page2.py::test_s25_combination_to_the_composition_then_a_wire_into_its_port (placed at once); test_gv_parity.py::test_todays_shortcuts (Ctrl+D at once) |
| test_a_file_added_by_mistake_is_warned_and_taken_out_one_by_one | PORTED | page::test_a_file_added_by_mistake_is_warned_and_taken_out_one_by_one |
| test_control_round_a_port_pulled_again_or_after_its_wire_went_says_so_or_gives_it_back | LOST | s6a-4 (the wire not given back); the rest: page::test_a_port_pulled_again_says_its_image_is_a_layer_already_and_a_shared_block_is_named_by_its_owner (the menu's order is GONE: the new menu) |
| test_control_round_a_branch_left_without_a_block_of_its_own_goes_with_it | LOST | s6a-20 |
| test_control_round_a_fusion_channel_sent_to_post_keeps_the_channels_apart | COVERED | tests/graphcore/test_core.py::test_layout_invariants_on_fixtures_and_folds (no block over another); flows::test_no_wire_of_the_templates_passes_under_a_block_nor_goes_right_to_left (the old lanes are GONE) |
| test_control_round_the_fusion_card_waits_for_the_channels_lights | PORTED | flows::test_a_combination_is_placed_empty_and_waits_for_its_wires_and_one_grid |
| test_control_round_undo_selects_the_block_before_and_a_file_taken_out_is_one_undo_step | COVERED | test_graphmodel_undo.py::test_undo_selects_on_block_ids; test_gv_page2.py::test_s33_files_filtered_by_type_take_all_one_undo_step |
| test_control_round_a_variant_of_a_graph_keeps_its_blocks_and_each_choice_is_said | COVERED | tests/graphcore/test_gestures3b.py::test_try_a_variant_then_keep_it_retires_the_original_id; Keep in the panel: model::test_try_as_variant_keeps_the_original_and_shares_the_blocks_before |
| test_f2_renames_and_ctrl_d_duplicates_what_is_selected | LOST | s6a-1 (F2); Ctrl+D: test_gv_parity.py::test_todays_shortcuts |
| test_rename_from_a_blocks_menu_types_the_name_on_the_block | LOST | s6a-1 |
| test_a_bare_image_workflow_is_renamed_on_its_block | LOST | s6a-1 |
| test_show_all_is_kept_with_the_project | COVERED | test_gv_sides.py::test_the_zoom_is_kept_per_project_in_the_settings |
| test_recombine_is_shown_greyed_before_the_lights | LOST | s6a-2 |
| test_a_chain_wire_says_it_cannot_be_removed | LOST | s6a-15 |
| test_the_palette_has_one_combination_entry_whose_types_unfold_beside_it | LOST | s6a-13 |
| test_a_linear_image_is_chosen_as_a_fusion_channel_on_its_row | PORTED | page::test_a_linear_image_is_chosen_as_a_fusion_channel_on_its_row |
| test_a_fusion_can_start_from_a_linear_image_alone | PORTED | page::test_a_fusion_can_start_from_a_linear_image_alone |
| test_a_linear_image_takes_stars_without_denoise | PORTED | page::test_a_linear_image_takes_stars_without_denoise |

## test_joins165.py (16)
| old test | verdict | where |
|---|---|---|
| test_the_sho_rgb_stars_template_makes_the_graph_of_the_issue | COVERED | test_graphmodel_facades.py::test_every_template_made_through_the_model_is_the_old_project[sho_rgb] (its Combine layers not drawn: s6a-2) |
| test_the_sho_rgb_template_shows_no_ghost_combine_layers_and_names_its_colour_workflow_rgb | COVERED | test_graphmodel_facades.py::test_every_template_made_through_the_model_is_the_old_project[sho_rgb] |
| test_a_combination_waiting_for_frames_is_greyed_never_red | PORTED | flows::test_a_combination_is_placed_empty_and_waits_for_its_wires_and_one_grid |
| test_the_names_of_the_inputs_never_sit_under_the_title_nor_the_subtitle | GONE | the old block item's layout of its inputs' names beside its title |
| test_the_wires_into_a_join_each_bend_at_their_own_place | GONE | the old router's staggered bends (`graph.stagger`) |
| test_no_wire_goes_right_to_left | COVERED | tests/graphcore/test_core.py::test_layout_invariants_on_fixtures_and_folds (I3); flows::test_no_wire_of_the_templates_passes_under_a_block_nor_goes_right_to_left |
| test_the_filters_templates_align_the_others_on_the_first | COVERED | test_graphmodel_facades.py::test_every_template_made_through_the_model_is_the_old_project[filters_full] |
| test_a_combination_placed_from_the_palette_is_empty_and_each_input_is_a_hollow_dot | PORTED | flows::test_a_combination_is_placed_empty_and_waits_for_its_wires_and_one_grid |
| test_a_wire_dropped_on_an_input_wires_it_one_step_each_and_undo_takes_it_back | COVERED | test_gv_page2.py::test_s25_combination_to_the_composition_then_a_wire_into_its_port |
| test_the_stars_alone_cannot_feed_a_combination_and_a_loop_is_refused | COVERED | test_gv_page2.py::test_s26_a_wire_closing_a_loop_turns_red_and_is_refused (the stars alone may feed a combination since the per-filter stars templates: GONE rule) |
| test_the_two_sides_of_combine_layers_are_wired_and_unwired | LOST | s6a-2 |
| test_the_names_are_french_with_aligner_sur | PORTED | misc::test_the_names_are_french_with_aligner_sur |
| test_a_join_stands_between_the_rows_it_reads_and_its_wires_never_cross | PORTED | flows::test_the_wires_into_a_join_never_cross (the join's height between its rows: the old layout, GONE) |
| test_the_filters_template_places_its_combination_level_with_the_middle_filter | PORTED | flows::test_the_wires_into_a_join_never_cross[filters_full] (level with the middle filter: the old layout, GONE) |
| test_every_filters_combination_takes_the_stars_off_per_filter_and_combines_them_like_the_nebula | PORTED | flows::test_every_filters_combination_takes_the_stars_off_per_filter_and_combines_them_like_the_nebula |
| test_the_dual_band_x2_template_makes_its_rows_and_its_combination | COVERED | test_graphmodel_facades.py::test_every_template_made_through_the_model_is_the_old_project[dual_x2] |

## test_align_block.py (9)
| old test | verdict | where |
|---|---|---|
| test_align_on_is_a_palette_block_placed_empty_and_greyed_with_its_reason | LOST | s6a-12 (an Image's Align on); a stack's "grid to choose": tests/graphcore/test_engine_rules.py::test_alignment_target_per_row |
| test_a_wire_dropped_on_the_reference_input_wires_it_in_one_undo_step | LOST | s6a-12 (on images); on stacks: test_gv_page.py::test_s12_reference_for_all_in_one_undo_step, flows::test_align_on_lists_every_other_stack_picks_one_refuses_a_loop_and_own_grid_undoes |
| test_a_real_drag_from_an_output_onto_the_reference_dot | LOST | s6a-12 (on images; on stacks the menu replaced the reference dot) |
| test_a_loop_is_refused_with_its_reason_and_nothing_changes | PORTED | flows::test_align_on_lists_every_other_stack_picks_one_refuses_a_loop_and_own_grid_undoes |
| test_the_panel_lists_every_workflow_for_the_reference_and_picks_it | PORTED | flows::test_align_on_lists_every_other_stack_picks_one_refuses_a_loop_and_own_grid_undoes (the stack's Align on menu) |
| test_the_cross_removes_the_block_and_ctrl_z_brings_it_back | PORTED | flows::test_align_on_lists_every_other_stack_picks_one_refuses_a_loop_and_own_grid_undoes (Own grid, Ctrl+Z) |
| test_align_images_is_gone_everywhere | PORTED | flows::test_align_images_is_gone_everywhere |
| test_inspect_reads_the_aligned_image_after_align_on | PORTED | flows::test_inspect_reads_the_aligned_image_after_align_on |
| test_plate_is_greyed_for_a_stack_with_its_reason_and_open_for_an_image | LOST | s6a-12 |

## test_blocks43_client.py (7)
| old test | verdict | where |
|---|---|---|
| test_two_sessions_are_drawn_before_the_chain_and_joined | COVERED | test_gv_page.py::test_s03_drag_session_onto_plus_session; tests/graphcore/test_points.py::test_a03_second_night_fans_in |
| test_the_session_block_edits_its_frames_and_its_filter | PORTED | flows::test_the_session_block_edits_its_frames_and_its_filter |
| test_switching_a_block_off_draws_it_so_refuses_what_cannot_and_undoes | COVERED | test_gv_page2.py::test_s20_d_switches_off_then_refuses_a_needed_block |
| test_a_workflow_from_an_image_starts_on_its_own_lane | LOST | s6a-8 (its line); the rest: flows::test_a_workflow_from_an_image_starts_on_its_own_row |
| test_filters_make_one_workflow_each_and_a_combination_in_the_composition_section | LOST | s6a-18 (the same filters again, from the New workflow window); the rest: flows::test_filters_make_one_workflow_each_and_their_combinations |
| test_an_already_stretched_image_workflow_goes_to_post | LOST | s6a-8 (its line); the rest: flows::test_an_already_stretched_image_workflow_goes_to_post |
| test_a_combination_panel_shows_its_named_inputs_and_its_settings | PORTED | flows::test_a_combination_panel_shows_its_named_inputs_and_its_settings |

## test_parallel_blocks.py (19)
| old test | verdict | where |
|---|---|---|
| test_a_card_that_can_only_be_set_in_parallel_is_available_by_a_drag | COVERED | test_gv_parity.py::test_a_card_refused_here_says_where_it_can_still_go |
| test_a_greyed_card_says_why_in_its_tooltip | COVERED | test_gv_page2.py::test_s24_palette_greys_a_card_and_says_why_on_hover |
| test_a_double_click_on_a_card_only_for_a_branch_places_nothing | COVERED | test_gv_parity.py::test_a_card_refused_here_says_where_it_can_still_go |
| test_the_places_are_shown_and_the_drop_makes_the_fork | COVERED | test_gv_page.py::test_s06_branch_strip_makes_a_variant_below |
| test_a_block_in_parallel_with_the_same_settings_computes_once_and_shows_on_its_own | PORTED | flows::test_a_variant_with_the_same_settings_computes_the_shared_block_once_and_keeps_its_own_settings |
| test_settings_of_a_block_in_parallel_stay_its_own | PORTED | flows::test_a_variant_with_the_same_settings_computes_the_shared_block_once_and_keeps_its_own_settings |
| test_a_second_calibration_forks_from_the_session | LOST | s6a-14 |
| test_its_files_are_changed_file_by_file_and_given_back | LOST | s6a-14 |
| test_its_files_are_listed_in_inspect_and_the_graph_panel_opens_it | LOST | s6a-14 |
| test_removed_with_nothing_else_its_branch_goes_and_the_trunk_stays | LOST | s6a-14 |
| test_a_branch_after_it_keeps_the_same_calibration_block | LOST | s6a-14 |
| test_on_several_nights_it_is_placed_under_one_night_the_others_stay_shared | LOST | s6a-14 |
| test_the_rows_of_the_nights_make_room | LOST | s6a-14 |
| test_its_files_on_a_night_and_its_removal | LOST | s6a-14 |
| test_on_several_nights_a_frame_block_in_parallel_starts_after_their_calibrations | COVERED | test_gv_page.py::test_s06_branch_strip_makes_a_variant_below (a variant from Registration starts after the nights' calibrations) |
| test_a_branch_of_a_branch_is_named_from_the_flow | PORTED | flows::test_a_variant_of_a_variant_has_a_name_of_its_own |
| test_inspect_lists_the_own_files_of_the_series_shown | LOST | s6a-14 |
| test_a_session_removed_takes_its_own_calibration_with_it | LOST | s6a-14 |
| test_nights_calibrated_together_say_what_to_do | COVERED | tests/graphcore/test_engine_rules.py::test_calibration_together_and_each_night |

## test_composition.py (12)
| old test | verdict | where |
|---|---|---|
| test_the_workflows_are_full_with_stars_and_classic_without | PORTED | flows::test_the_layers_post_gives_a_stars_workflow_and_another |
| test_a_stars_workflow_comes_as_its_stars_and_background_block_and_the_others_as_their_output | LOST | s6a-2 (no Combine layers block); the layers to come: flows::test_the_layers_post_gives_a_stars_workflow_and_another |
| test_the_default_layer_of_a_stars_workflow_puts_the_stars_back_in_screen | PORTED | misc::test_the_default_layer_of_a_stars_workflow_puts_the_stars_back_in_screen |
| test_the_model_adds_changes_and_removes_a_post_layer_and_says_who_did | PORTED | flows::test_the_model_adds_changes_and_removes_a_post_layer_and_says_who_did |
| test_removing_a_layer_takes_its_own_filters_with_it | PORTED | flows::test_removing_a_layer_takes_its_own_filters_with_it |
| test_the_graph_has_a_block_per_stars_and_background_layer_and_sends_the_others_as_they_are | LOST | s6a-2 (the layers' names: flows::test_the_model_adds_changes_and_removes_a_post_layer_and_says_who_did) |
| test_an_image_layer_counts_in_the_post_block_without_a_workflow | LOST | s6a-2 (the Post block's line); Post's layers: flows::test_an_image_layer_counts_among_post_s_layers |
| test_a_part_layer_asks_the_stars_block_for_its_images | PORTED | misc::test_a_part_layer_asks_the_stars_block_for_its_images |
| test_load_keeps_a_valid_blend_and_drops_an_unknown_part | PORTED | misc::test_load_keeps_a_valid_blend_and_drops_an_unknown_part |
| test_linear_dodge_adds_the_layers | PORTED | misc::test_linear_dodge_adds_the_layers |
| test_the_stars_panel_sends_each_image_to_post_and_makes_a_stars_and_background_block | PORTED | flows::test_the_stars_panel_sends_each_image_to_post_and_the_stars_alone_can_be_taken_back |
| test_sending_the_stars_alone_keeps_the_combine_block_and_the_link_can_be_taken_back | PORTED | flows::test_the_stars_panel_sends_each_image_to_post_and_the_stars_alone_can_be_taken_back (the graph's Combine layers block: s6a-2) |

## test_fusion_sessions_client.py (15)
| old test | verdict | where |
|---|---|---|
| test_a_second_session_joins_the_chosen_flow_never_in_place_of_its_nights | PORTED | flows::test_a_second_session_joins_the_chosen_stack_never_in_place_of_its_nights |
| test_a_session_joining_a_filter_flow_takes_its_filter_and_a_fusion_chosen_starts_a_flow | LOST | s6a-9 (the filter not taken); a combination chosen: flows::test_a_session_chosen_with_a_combination_starts_a_stack_of_its_own |
| test_a_session_added_to_a_filter_flow_follows_the_filter_of_its_frames | PORTED | flows::test_a_session_added_to_a_filter_flow_follows_the_filter_of_its_frames |
| test_a_combination_is_placed_empty_and_waits_for_its_wires_and_one_grid | PORTED | flows::test_a_combination_is_placed_empty_and_waits_for_its_wires_and_one_grid |
| test_one_image_sent_to_post_from_a_block_is_not_numbered | GONE | an Image block's title is its kind; the workflow's name is its row's caption (page::test_images_are_numbered_by_their_workflows) |
| test_a_stars_block_after_a_combination_is_drawn_after_it | COVERED | tests/graphcore/test_engine_rules.py::test_after_a_combination_only_stars_and_color_and_nothing_after_combine_layers |
| test_a_fusion_input_keeps_its_workflow | LOST | s6a-10 |
| test_two_images_without_a_file_are_two_blocks | PORTED | page::test_two_images_without_a_file_are_two_blocks |
| test_names_never_empty_nor_another_s | PORTED | flows::test_names_are_never_empty_nor_another_s (the name typed on a block: s6a-1) |
| test_removing_a_flow_or_its_session_says_what_goes | LOST | s6a-10 (the Session left); the rest: flows::test_removing_a_stack_or_its_session_says_what_goes |
| test_a_session_calibration_shows_its_scope_and_has_its_buttons | LOST | s6a-5 (a pause on Calibration said, not kept); the rest: flows::test_a_session_calibration_shows_its_scope_and_has_its_buttons |
| test_a_field_going_away_writes_nothing | PORTED | misc::test_a_field_going_away_writes_nothing |
| test_a_filter_s_second_night_is_named_and_drawn_with_its_filter | LOST | s6a-9 |
| test_a_new_session_s_name_is_never_another_s | PORTED | flows::test_a_new_session_s_name_is_never_another_s |
| test_the_deconvolution_and_synthetic_stars_cards_get_their_durations | PORTED | flows::test_the_deconvolution_and_synthetic_stars_cards_get_their_durations |

## test_graph_wires_clear.py (2)
| old test | verdict | where |
|---|---|---|
| test_no_wire_of_the_templates_passes_under_a_block | PORTED | flows::test_no_wire_of_the_templates_passes_under_a_block_nor_goes_right_to_left |
| test_synthetic_stars_feed_combine_layers_and_no_wire_passes_under_it | LOST | s6a-2 (no Combine layers to feed); no wire under a block: flows::test_synthetic_stars_after_stars_and_background_draw_no_output_block_and_no_wire_under_a_block |

## test_no_output_block.py (4)
| old test | verdict | where |
|---|---|---|
| test_no_template_and_no_block_placed_after_it_draws_an_output_block | PORTED | flows::test_no_template_and_no_block_placed_after_it_draws_an_output_block |
| test_synthetic_stars_after_stars_and_background_draw_no_output_block | PORTED | flows::test_synthetic_stars_after_stars_and_background_draw_no_output_block_and_no_wire_under_a_block |
| test_images_and_their_blocks_draw_no_output_block | PORTED | flows::test_images_and_their_blocks_draw_no_output_block |
| test_filter_workflows_and_their_fusion_draw_no_output_block | PORTED | flows::test_filter_workflows_and_their_combination_draw_no_output_block |

## test_waiting_wire.py (5)
| old test | verdict | where |
|---|---|---|
| test_a_workflow_without_lights_has_a_dashed_wire_and_no_layer | LOST | s6a-3 |
| test_to_post_on_the_last_block_says_lights_first_not_already_a_layer | LOST | s6a-3 |
| test_lights_turn_the_dashed_wire_into_a_layer | PORTED | flows::test_with_lights_a_workflow_s_wire_to_post_is_a_layer |
| test_every_block_says_what_it_does | LOST | s6a-17 (the line not wrapped); what each says: flows::test_every_block_says_what_it_does |
| test_a_banner_stays_until_it_is_closed | PORTED | misc::test_a_banner_stays_until_it_is_closed |
