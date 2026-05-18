u# LIBERO (SFT baseline)
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi0_fast_libero_low_mem_finetune --policy.dir ./checkpoints/pi0_fast_libero_icl_lora"
export CLIENT_ARGS="--args.task-suite-name libero_spatial --args.task-ids 8,9"
MUJOCO_GL=egl docker compose -f examples/libero/compose.yml up --build

# LIBERO RICL
export RICL_CONFIG=pi0_fast_libero_ricl_low_mem
export RICL_CHECKPOINT=/app/checkpoints/pi0_fast_libero_icl_priming_lora_ckpt_4500
export RICL_DEMOS=/app/ricl_libero_preprocessing/collected_demos/libero_group/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
# IMPORTANT: Use the max_distance.json computed from LIBERO training data, NOT the DROID one!
export RICL_MAX_DIST=/app/preprocessing/collected_demos_training/max_distance.json
# Note: ricl_step_offset is now baked into the model config (e.g. pi0_fast_libero_ricl_step5_offset),
# so just point RICL_CONFIG / RICL_CHECKPOINT at the right ckpt — no separate inference flag.
docker compose -f examples/libero/compose.yml -f examples/libero/compose.ricl.yml up --build