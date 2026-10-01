#!/bin/bash -l

export TORCH_DISTRIBUTED_DEBUG=INFO

start_date=$(date)

work_root="${POSEDRIVER_WORKDIR:?Set POSEDRIVER_WORKDIR to a writable experiment directory}"
cd ${work_root}

number_gpus=$(nvidia-smi --list-gpus | wc -l)

project_name='opp_hrformer'
script_name=$(basename "$0" .sh)

datasets_folder="${POSEDRIVER_DATA_ROOT:?Set POSEDRIVER_DATA_ROOT to the prepared datasets directory}"
dataset='animal_keypoint_datasets/AwA_animalpose'

model="${work_root}/results/cocoani-convnextv2base-l200-mixedbs64-8-all/checkpoints/cocoani-convnextv2base-l200-mixedbs64-8-all.pt.epoch450"
epochs=550

# results_folder=${work_root}/results/${script_name}
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
#pip install opencv-python-headless mmengine mmcv mmpose 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
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
  --dataset=cocobicyclekp \
  --cocobicyclekp-train-image-dir=${datasets_folder}/cocobicycle/splits/fold_1_train \
  --cocobicyclekp-train-annotations=${datasets_folder}/cocobicycle/splits/modified_fold_1_train.json \
  --cocobicyclekp-val-image-dir=${datasets_folder}/cocobicycle/splits/fold_1_val \
  --cocobicyclekp-val-annotations=${datasets_folder}/cocobicycle/splits/modified_fold_1_val.json \
  --cocobicyclekp-square-edge=513 \
  --cocobicyclekp-extended-scale \
  --cocobicyclekp-orientation-invariant=0.1 \
  --cocobicyclekp-upsample=2 \
  ${init_args} \
  --adamw \
  --epochs=${epochs} \
  --batch-size=8 \
  --lr=0.001 \
  --lr-decay-type=linear \
  --lr-warm-up-type=linear \
  --lr-warm-up-epochs=20 \
  --lr-warm-up-start-epoch=450 \
  --momentum=0.9 \
  --weight-decay=0.1 \
  --b-scale=10.0 \
  --clip-grad-value=10.0 \
  2>&1 | tee -a ${results_folder}/logs/train__${script_name}__${cont_ext}
echo "Done training" 2>&1 | tee -a ${results_folder}/logs/train__${script_name}__${cont_ext}

# Validation
eval_ckpt="${results_folder}/checkpoints/${script_name}.pt.epoch$(printf "%03d" ${epochs})"
echo "eval_ckpt=${eval_ckpt}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
echo "Start evaluating ${eval_ckpt}..." 2>&1 | tee ${results_folder}/logs/val_epoch${epochs}__${script_name}__${cont_ext}
CUDA_VISIBLE_DEVICES=0 time python3 -m openpifpaf.eval \
  --write-predictions \
  --output=${results_folder}/predictions/val_epoch${epochs}__${script_name}__${cont_ext} \
  --dataset=cocobicyclekp \
  --cocobicyclekp-train-image-dir=${datasets_folder}/cocobicycle/splits/fold_1_train \
  --cocobicyclekp-train-annotations=${datasets_folder}/cocobicycle/splits/modified_fold_1_train.json \
  --cocobicyclekp-val-image-dir=${datasets_folder}/cocobicycle/splits/fold_1_val \
  --cocobicyclekp-val-annotations=${datasets_folder}/cocobicycle/splits/modified_fold_1_val.json \
  --cocobicyclekp-upsample=2 \
  --cocobicyclekp-no-eval-annotation-filter \
  --batch-size=1 \
  --loader-workers=8 \
  --checkpoint=${eval_ckpt} \
  --decoder=cifcaf:0 \
  --seed-threshold=0.2 \
  --force-complete-pose \
  2>&1 | tee -a ${results_folder}/logs/val_epoch${epochs}__${script_name}__${cont_ext}
echo "Done evaluating ${eval_ckpt}" 2>&1 | tee -a ${results_folder}/logs/val_epoch${epochs}__${script_name}__${cont_ext}

end_date=$(date)
echo "Ending date: ${end_date}" 2>&1 | tee -a ${results_folder}/logs/setup__${cont_ext}
