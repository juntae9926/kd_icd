MASTER_ADDR="localhost" MASTER_PORT="20281" NODE_RANK="0" WORLD_SIZE=2 \
  ./sscd/train.py --nodes=1 --gpus=2 \
  --train_dataset_path /dataset/DISC/images/dev_queries \
  --val_dataset_path /dataset/DISC/val_images \
  --query_dataset_path /dataset/DISC/images/final_queries \
  --ref_dataset_path /dataset/DISC/images/references \
  --augmentations=ADVANCED --mixup=true \
  --batch_size 256 \
  --workers 16 \
  --absolute_learning_rate 0.05 \
  --output_path=./ \
  --ckpt=./weights/sscd_disc_mixup.torchvision.pt \
  --backbone=TV_MOBILENETV3 \
  --kd True