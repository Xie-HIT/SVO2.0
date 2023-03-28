#!/bin/bash
# evaluate SVO2.0 in Newer College dataset (Collection 1)
# @author xie chen
# @time 2023/02/23

:<<!
output directory will be arranged as follows:
├── Quad-Easy
│   ├── Evaluate-Quad-Easy.bag
│   ├── stamped_groundtruth.txt
│   └── stamped_traj_estimate.txt
├── Quad-Medium
│   ├── Evaluate-Quad-Hard.bag
│   ├── stamped_groundtruth.txt
│   └── stamped_traj_estimate.txt
└── Quad-Hard
    ├── Evaluate-Quad-Medium.bag
    ├── stamped_groundtruth.txt
    └── stamped_traj_estimate.txt
!
clear

DATASET_ROOT="/home/xgrids/SLAM数据集/Newer_College/Collection1"
EVALUATE_OUTPUT_DIR="/home/xgrids/svo_ws/src/test/Newer_College/test1"

help(){
  echo "使用:  ./evaluate.sh [OPTION…]"
  echo "参数:  "
  echo "       -h    Help"
  echo "       -e    Don't run dataset, just try to evaluate"
  echo "       -r    Remove exist folder and re-run"
  exit
}

while getopts :her OPTION;do
    case $OPTION in
    h)help
    ;;
    e)SKIP=true
    ;;
    r)REMOVE=true
    ;;
    ?)help
    ;;
    esac
done

# create root directory
echo "-- Evaluating on Newer College datasets"
if ! [[ -e $EVALUATE_OUTPUT_DIR ]]
then
  mkdir -p $EVALUATE_OUTPUT_DIR
fi

# check dataset exist and run them
for var in "Quad-Hard" # "Quad-Medium" "Quad-Hard"
do
  WORK_DIR=$EVALUATE_OUTPUT_DIR/$var

  if [[ -e "$DATASET_ROOT""/$var" ]]
  then
    printf "\033[32m [Success] Found dataset: %s \033[0m\n" "$DATASET_ROOT""/$var"
    if ! [[ -e $WORK_DIR ]]
    then
      mkdir -p $WORK_DIR
      echo "-- Build new folder: $WORK_DIR"
    elif [ $REMOVE ]
    then
      rm -rf $WORK_DIR
      echo "-- Remove old folder: $WORK_DIR"
      mkdir -p $WORK_DIR
      echo "-- Build new folder: $WORK_DIR"
    fi
  else
    printf "\033[33m [Warning] No such dataset: %s \033[0m\n" "$DATASET_ROOT""/$var"
  fi

  echo "-- Running SLAM on $var..."

if ! [ $SKIP ]
then
  gnome-terminal --tab  -q --command="bash -c 'roslaunch svo_ros newer_college_vio_multicam.launch; $SHELL'" \
  --tab -q --command="bash -c 'cd /home/xgrids/SLAM数据集/Newer_College/Collection1/$var; rosbag play -r 1 -d 10 $var.bag'" \
  --tab -q --command="bash -c 'cd $WORK_DIR; rosbag record --output-name=Evaluate-$var /svo/pose_imu __name:=my_bag; rosnode kill /my_bag'"
fi

  # wait SLAM to finish
  continue="n"
  while ! [[ $continue == Y || $continue == y ]]
  do
    read -a continue -rp "-- Continue(Y/n): "
    if ! [[ ${continue[0]} == Y || ${continue[0]} == N || ${continue[0]} == y || ${continue[0]} == n ]]
    then
      continue
    elif [[ ${continue[0]} == N || ${continue[0]} == n ]]
    then
      exit
    fi
  done

  echo "-- Evaluating on $var..."
  sleep 5s

  # transform to TUM format, and transform pose from imu coordinate to base coordinate
  if ! [ -e $WORK_DIR/stamped_traj_estimate.txt ]
  then
    rosrun rpg_trajectory_evaluation bag_to_pose.py \
        $WORK_DIR/Evaluate-$var.bag \
        /svo/pose_imu --out=stamped_traj_estimate.txt

    mv $WORK_DIR/stamped_traj_estimate.txt $WORK_DIR/stamped_traj_estimate_bak.txt
    source ~/anaconda3/etc/profile.d/conda.sh
    python3 ./transform_to_imu.py \
    --input $WORK_DIR/stamped_traj_estimate_bak.txt \
    --output $WORK_DIR/stamped_traj_estimate.txt
    rm $WORK_DIR/stamped_traj_estimate_bak.txt
  fi

  # copy groundtruth to evaluate directory
  if ! [ -e $WORK_DIR/stamped_groundtruth.txt ]
  then
    cp $EVALUATE_OUTPUT_DIR/../groundtruth/$var/stamped_groundtruth.txt $WORK_DIR
  fi

  # evaluate
  cd $WORK_DIR || exit; evo_ape tum stamped_groundtruth.txt stamped_traj_estimate.txt -vap --n_to_align 100
  cd $WORK_DIR || exit; evo_rpe tum stamped_groundtruth.txt stamped_traj_estimate.txt -r full -a --n_to_align 100 --delta 1 --plot --plot_mode xyz
  cd $WORK_DIR || exit; evo_traj tum --ref stamped_groundtruth.txt stamped_traj_estimate.txt -ap --n_to_align 100
done