MASTER_ADDR="localhost" MASTER_PORT="20000" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train.py --nodes=1 --gpus=2 \
  --train_dataset_path {YOUR_TRAIN_PATH} \
  --val_dataset_path {YOUR_VALIDATION_PATH} \
  --query_dataset_path {YOUR_QUERY_PATH} \
  --ref_dataset_path {YOUR_REFERENCE_PATH} \
  --augmentations=ADVANCED --mixup=true \
  --batch_size 256 \
  --workers 16 \
  --absolute_learning_rate 0.05 \
  --output_path=./ \
  --ckpt=./weights/sscd_disc_mixup.torchvision.pt \
  --backbone=TV_MOBILENETV3 \
  --kd True
