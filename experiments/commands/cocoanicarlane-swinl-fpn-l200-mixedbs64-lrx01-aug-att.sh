#!/bin/bash -l

export TORCH_DISTRIBUTED_DEBUG=INFO

start_date=$(date)

work_root="${POSEDRIVER_WORKDIR:?Set POSEDRIVER_WORKDIR to a writable experiment directory}"
cd ${work_root}

number_gpus=$(nvidia-smi --list-gpus | wc -l)

project_name='opp_hrformer'
script_name=$(basename "$0" .sh)

datasets_folder="${POSEDRIVER_DATA_ROOT:?Set POSEDRIVER_DATA_ROOT to the prepared datasets directory}"
lane_annotations="${POSEDRIVER_LANE_ANNOT_ROOT:-${work_root}/openlane_annot}"
dataset='MS_COCO+Animal-Pose'
dataset1='MS_COCO'
dataset22='Animal-Pose'
dataset2='animal_keypoint_datasets/AwA_animalpose'
dataset3='apolloscape'

model="${work_root}/results/cocokp-swinl-fpn-250epochs-lrx01-att/checkpoints/cocokp-swinl-fpn-250epochs-lrx01-att.pt.epoch250"
epochs=450

results_folder=${work_root}/results/${script_name}

# Check if new or resuming run
if [ -d "${results_folder}" ]; then
  echo "Resuming existing experiment from ${results_folder}"
  net_ckpt=$(ls ${results_folder}/checkpoints/${script_name}.pt.epoch* | sort -V | tail -n 1)
  optim_ckpt=$(ls ${results_folder}/checkpoints/${script_name}.pt.optim.epoch* | sort -V | tail -n 1)
  init_args="--checkpoint=${net_ckpt} --resume-training=${optim_ckpt}"
  previous_run_log=$(ls ${results_folder}/logs/setup__run* | sort -V | tail -n 1)
  previous_run_name=$(basename ${previous_run_log})
  previous_run_id=${previous_run_name: -3}
  run_id=$((previous_run_id+1))
  cont_ext="run$(printf "%03d" ${run_id})"
else
  echo "Starting new experiment in ${results_folder}"
  mkdir -p ${results_folder}
  mkdir ${results_folder}/logs
  mkdir ${results_folder}/checkpoints
  mkdir ${results_folder}/predictions
  init_args="--checkpoint=${model}"
  cont_ext='run000'
fi

# Log setup and config
touch ${results_folder}/logs/setup__${cont_ext}
echo "Starting date: ${start_date}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "Setup:" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "home=${work_root}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "number_gpus=${number_gpus}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "project_name=${project_name}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "script_name=${script_name}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "datasets_folder=${datasets_folder}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "dataset=${dataset}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "model=${model}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "epochs=${epochs}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "results_folder=${results_folder}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "cont_ext=${cont_ext}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "init_args=${init_args}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}

# Install OpenPifPaf
#echo "Install OpenPifPaf with pip install ${work_root}/projects/${project_name}/openpifpaf[backbones,dev,test,train]" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
#pip install --root-user-action=ignore --upgrade pip wheel setuptools 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
#pip install ${work_root}/projects/${project_name}/openpifpaf[backbones,dev,test,train] 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
#pip install --no-build-isolation ${work_root}/projects/${project_name}/openpifpaf[backbones,dev,test,train] 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
pip list 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}

# Training
echo "Start training..." 2>&1 | tee ${results_folder}/logs/train__${script_name}__${cont_ext}
if [ "${number_gpus}" -gt "1" ]; then
  train_command="torchrun -m --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node=${number_gpus} openpifpaf.train --ddp"
else
  train_command="python3 -m openpifpaf.train"
fi
time ${train_command} \
  --output=${results_folder}/checkpoints/${script_name}.pt \
  --dataset=cocokp-animal-apollo-openlane \
  --cocokp-train-image-dir=${datasets_folder}/${dataset1}/images/train_aug/ \
  --cocokp-train-annotations=${datasets_folder}/${dataset1}/annotations/person_keypoint_aug4.json \
  --cocokp-val-image-dir=${datasets_folder}/${dataset1}/images/val2017/ \
  --cocokp-val-annotations=${datasets_folder}/${dataset1}/annotations/person_keypoints_val2017.json \
  --cocokp-square-edge=513 \
  --cocokp-extended-scale \
  --cocokp-orientation-invariant=0.1 \
  --cocokp-upsample=2 \
  --animal-train-image-dir=${datasets_folder}/${dataset2}/images/train_aug/ \
  --animal-train-annotations=${datasets_folder}/${dataset2}/annotations/animal_keypoint_aug4.json \
  --animal-val-image-dir=${datasets_folder}/${dataset2}/images/val/ \
  --animal-val-annotations=${datasets_folder}/${dataset2}/annotations/val_animalpose.json \
  --animal-square-edge=385 \
  --animal-extended-scale \
  --animal-orientation-invariant=0.1 \
  --animal-upsample=2 \
  --animal-bmin=2 \
  --apollo-train-image-dir=${datasets_folder}/${dataset3}/images/train_aug_24/ \
  --apollo-val-image-dir=${datasets_folder}/${dataset3}/images/val/ \
  --apollo-train-annotations=${datasets_folder}/${dataset3}/annotations/apolloscape_keypoint_aug4.json \
  --apollo-val-annotations=${datasets_folder}/${dataset3}/annotations/apollo_keypoints_24_val.json \
  --apollo-upsample=2 \
  --apollo-square-edge=513 \
  --apollo-extended-scale \
  --apollo-use-24-kps \
  --openlane-train-image-dir=${datasets_folder}/OpenLane/OpenDriveLab___OpenLane/raw/images/training/ \
  --openlane-train-annotations=${lane_annotations}/openlane_keypoints_training_2.json \
  --openlane-val-image-dir=${datasets_folder}/OpenLane/OpenDriveLab___OpenLane/raw/images/validation/ \
  --openlane-val-annotations=${lane_annotations}/openlane_keypoints_validation_2.json \
  --dataset-weights 0.5 1.0 1.0 1.0 \
  --cf4-attention-swin-head True \
  ${init_args} \
  --swin-use-fpn \
  --adamw \
  --epochs=${epochs} \
  --batch-size=16 \
  --stride-apply=2 \
  --lr=0.00005 \
  --lr-decay-type=linear \
  --lr-warm-up-type=linear \
  --lr-warm-up-epochs=20 \
  --lr-warm-up-start-epoch=250 \
  --momentum=0.9 \
  --weight-decay=0.1 \
  --b-scale=10.0 \
  --clip-grad-value=10.0 \
  2>&1 | tee -a ${results_folder}/logs/train__${script_name}__${cont_ext}
echo "Done training" 2>&1 | tee -a ${results_folder}/logs/train__${script_name}__${cont_ext}

# Validation
eval_ckpt="${results_folder}/checkpoints/${script_name}.pt.epoch$(printf "%03d" ${epochs})"
echo "eval_ckpt=${eval_ckpt}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "Start evaluating ${eval_ckpt} on ${dataset1}..." 2>&1 | tee ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset1}__${cont_ext}
CUDA_VISIBLE_DEVICES=0 time python3 -m openpifpaf.eval \
  --write-predictions \
  --output=${results_folder}/predictions/val_epoch${epochs}__${script_name}__${dataset1}__${cont_ext} \
  --dataset=cocokp \
  --cocokp-train-image-dir=${datasets_folder}/${dataset1}/images/train2017/ \
  --cocokp-train-annotations=${datasets_folder}/${dataset1}/annotations/person_keypoints_train2017.json \
  --cocokp-val-image-dir=${datasets_folder}/${dataset1}/images/val2017/ \
  --cocokp-val-annotations=${datasets_folder}/${dataset1}/annotations/person_keypoints_val2017.json \
  --cocokp-square-edge=641 \
  --cocokp-upsample=2 \
  --coco-no-eval-annotation-filter \
  --cf4-attention-swin-head True \
  --batch-size=1 \
  --loader-workers=8 \
  --checkpoint=${eval_ckpt} \
  --swin-use-fpn \
  --decoder=cifcaf:0 \
  --seed-threshold=0.2 \
  --force-complete-pose \
  2>&1 | tee -a ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset1}__${cont_ext}
echo "Done evaluating ${eval_ckpt} on ${dataset1}" 2>&1 | tee -a ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset1}__${cont_ext}
echo "Start evaluating ${eval_ckpt} on ${dataset2}..." 2>&1 | tee ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset22}__${cont_ext}
CUDA_VISIBLE_DEVICES=0 time python3 -m openpifpaf.eval \
  --write-predictions \
  --output=${results_folder}/predictions/val_epoch${epochs}__${script_name}__${dataset22}__${cont_ext} \
  --dataset=animal \
  --animal-train-image-dir=${datasets_folder}/${dataset2}/images/train/ \
  --animal-train-annotations=${datasets_folder}/${dataset2}/annotations/train_awa_animalpose.json \
  --animal-val-image-dir=${datasets_folder}/${dataset2}/images/val/ \
  --animal-val-annotations=${datasets_folder}/${dataset2}/annotations/val_animalpose.json \
  --animal-upsample=2 \
  --animal-no-eval-annotation-filter \
  --cf4-attention-swin-head True \
  --batch-size=1 \
  --loader-workers=8 \
  --checkpoint=${eval_ckpt} \
  --swin-use-fpn \
  --decoder=cifcaf:0 \
  --seed-threshold=0.01 \
  --force-complete-pose \
  2>&1 | tee -a ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset22}__${cont_ext}
echo "Done evaluating ${eval_ckpt} on ${dataset22}" 2>&1 | tee -a ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset22}__${cont_ext}
echo "Start evaluating ${eval_ckpt} on ${dataset3}..." 2>&1 | tee ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset3}__${cont_ext}
CUDA_VISIBLE_DEVICES=0 time python3 -m openpifpaf.eval \
  --write-predictions \
  --output=${results_folder}/val_epoch${epochs}__${script_name}__${dataset3}__${cont_ext} \
  --dataset=apollo \
  --apollo-train-image-dir=${datasets_folder}/${dataset3}/images/train/ \
  --apollo-train-annotations=${datasets_folder}/${dataset3}/annotations/apollo_keypoints_24_train.json \
  --apollo-val-image-dir=${datasets_folder}/${dataset3}/images/val/ \
  --apollo-val-annotations=${datasets_folder}/${dataset3}/annotations/apollo_keypoints_24_val.json \
  --apollo-use-24-kps \
  --apollo-upsample=2 \
  --apollo-no-eval-annotation-filter \
  --cf4-attention-swin-head True \
  --batch-size=1 \
  --loader-workers=8 \
  --checkpoint=${eval_ckpt} \
  --swin-use-fpn \
  --decoder=cifcaf:0 \
  --seed-threshold=0.2 \
  --force-complete-pose \
  2>&1 | tee -a ${results_folder}/logs/val_average_${dataset3}
echo "Done evaluating ${eval_ckpt} on ${dataset3}" 2>&1 | tee -a ${results_folder}/logs/val_epoch${epochs}__${script_name}__${dataset3}__${cont_ext}


end_date=$(date)
echo "Ending date: ${end_date}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
