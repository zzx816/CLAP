# Run from /home/runner/work/CLAP/CLAP/src
# Example:
# cd /home/runner/work/CLAP/CLAP/src
# CUDA_VISIBLE_DEVICES=0 python -m evaluate.eval_linear_probe \
#   ... (same args as below)

python -m evaluate.eval_linear_probe \
  --save-frequency 10 \
  --save-top-performance 3 \
  --save-most-recent \
  --dataset-type="webdataset" \
  --datasetpath="/path/to/your/webdataset_tar" \
  --precision="fp32" \
  --warmup 0 \
  --batch-size=64 \
  --lr=1e-4 \
  --wd=0.1 \
  --epochs=30 \
  --workers=4 \
  --freeze-text \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --datasetnames "underwater_target" \
  --datasetinfos "train" \
  --seed 3407 \
  --logs "/path/to/your/logs" \
  --gather-with-grad \
  --lp-loss="ce" \
  --lp-metrics="acc" \
  --lp-lr=1e-4 \
  --lp-mlp \
  --class-label-path="../class_labels/Underwater_class_labels_indices_space_template.json" \
  --pretrained="/path/to/your/checkpoint_dir" \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --optimizer "adam"
