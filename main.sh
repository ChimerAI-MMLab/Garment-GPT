export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

python main.py \
  --llm_model_path "checkpoints/vlm/checkpoint-12844" \
  --codec_config_path "configs/config_vq1024_resres_aug_decay0.99_q5_gcd_nl8_ld512.yaml" \
  --rt_config_path "configs/config_rt_euler.yaml" \
  --image_path "rand_YHLMQOMVPI_render_front_cutted.png" \
  --output_path "./my_garment.json" \
  --device "cuda:0"
