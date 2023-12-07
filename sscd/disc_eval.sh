sscd/disc_eval.py --disc_path /dataset/DISC --gpus=2 \
  --output_path ./disc_eval/efficientnet \
  --size 288 \
  --preserve_aspect_ratio false \
  --workers=28 \
  --backbone TV_EFFICIENTNET_B0 \
  --dims 512 \
  --student /user/weights/efficientnet.ckpt
