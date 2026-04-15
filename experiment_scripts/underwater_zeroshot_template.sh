# Run from /home/runner/work/CLAP/CLAP/src
# Example:
# cd /home/runner/work/CLAP/CLAP/src
# CUDA_VISIBLE_DEVICES=0 bash ../experiment_scripts/underwater_zeroshot_template.sh

python -m evaluate.eval_zeroshot_classification \
  --dataset-type="webdataset" \
  --datasetpath="/path/to/your/webdataset_tar" \
  --precision="fp32" \
  --batch-size=128 \
  --workers=4 \
  --amodel HTSAT-tiny \
  --tmodel roberta \
  --datasetnames "underwater_target" \
  --datasetinfos "train" \
  --seed 3407 \
  --logs "/path/to/your/logs" \
  --data-filling "repeatpad" \
  --data-truncating "rand_trunc" \
  --freeze-text \
  --class-label-path="../class_labels/Underwater_class_labels_indices_space_template.json" \
  --pretrained="/path/to/your/checkpoint/or/checkpoint_dir"
