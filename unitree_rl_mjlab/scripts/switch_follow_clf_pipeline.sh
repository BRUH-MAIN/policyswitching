#!/bin/bash
# Collect scans -> train classifiers -> calibrate the switch filter (seeds 400/401).
set -u
W=/home/rohan/rl/policyswitching/.claude/worktrees/vlm-pipeline/unitree_rl_mjlab
PY=/home/rohan/miniconda3/envs/unitree_rl_mjlab/bin/python
CK=/home/rohan/rl/policyswitching/unitree_rl_mjlab/eval_ckpts
L=logs/switch_follow  # scans and classifier weights (gitignored: regenerate with this script)
cd $W
export PYTHONPATH=$W MUJOCO_GL=egl
until [ -f $L/confirm_noise_s502.log ] && [ "$(wc -l < $L/confirm_noise_s502.log)" -ge 10 ]; do sleep 20; done
for cond in noisy clean; do
  flag=""; [ $cond = noisy ] && flag="--obs-noise"
  for course in rough stairs_up stairs_down; do
    for s in 700 701; do
      rm -f $L/collect_${cond}_${course}_s$s.json
      $PY scripts/switch_follow.py --course $course --seed $s --num-envs 128 --arms hard:0.3:label $flag \
        --record-scans $L/scans_${cond}_${course}_s$s.pt --ckpt-root $CK \
        --json-out $L/collect_${cond}_${course}_s$s.json 2>&1 | grep -a 'RESULT\|INFO\] wrote\|Error\|Traceback\|  File' | tee -a $L/clf_pipeline.log
    done
  done
  $PY scripts/scan_classifier_train.py $L/scans_${cond}_*.pt --out $L/clf_${cond}.pt > $L/clf_${cond}_train.json 2>> $L/clf_pipeline.log
  echo "[TRAINED] $cond $(grep val_accuracy $L/clf_${cond}_train.json)" | tee -a $L/clf_pipeline.log
done
for cond in noisy clean; do
  flag=""; [ $cond = noisy ] && flag="--obs-noise"
  ARMS="hard:0.3:label"
  for f in 1.0:1 0.3:3 0.1:5 0.05:10; do ARMS="$ARMS clf:$L/clf_${cond}.pt:$f"; done
  for s in 400 401; do
    $PY scripts/switch_follow.py --seed $s --num-envs 256 --arms $ARMS $flag --resume --ckpt-root $CK \
      --json-out eval_results/switch_follow/clf_calib_${cond}_s$s.json 2>&1 | grep -a 'RESULT\|SKIP\|Error\|Traceback\|  File' | tee -a $L/clf_pipeline.log
  done
done
echo CLF_PIPELINE_DONE | tee -a $L/clf_pipeline.log
