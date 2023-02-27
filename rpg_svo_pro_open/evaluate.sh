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
  exit
}

while getopts :he OPTION;do
    case $OPTION in
    h)help
    ;;
    e)SKIP=true
    ;;
    ?)help
    ;;
    esac
done

# create root directory
echo "-- Evaluating on Newer College datasets"
if ! [[ -e $EVALUATE_OUTPUT_DIR ]]
then
  mkdir $EVALUATE_OUTPUT_DIR
fi

# check dataset exist and run them
for var in "Quad-Easy" # "Quad-Medium" "Quad-Hard"
do
  WORK_DIR=$EVALUATE_OUTPUT_DIR/$var

  if [[ -e "$DATASET_ROOT""/$var" ]]
  then
    printf "\033[32m [Success] Found dataset: %s \033[0m\n" "$DATASET_ROOT""/$var"
    if ! [[ -e WORK_DIR ]]
    then
      mkdir $WORK_DIR
    fi
    echo "Output will be put on: $WORK_DIR"
  else
    printf "\033[33m [Warning] No such dataset: %s \033[0m\n" "$DATASET_ROOT""/$var"
  fi

  echo "Running SLAM on $var..."

if ! [ $SKIP ]
then
  gnome-terminal --tab  -q --command="bash -c 'roslaunch svo_ros newer_college_vio_multicam.launch'" \
  --tab -q --command="bash -c 'cd /home/xgrids/SLAM数据集/Newer_College/Collection1/$var; rosbag play -d 10 $var.bag'" \
  --tab -q --command="bash -c 'cd $WORK_DIR; rosbag record --output-name=Evaluate-$var /svo/pose_imu __name:=my_bag; rosnode kill /my_bag'"
fi

  # wait SLAM to finish
  continue=n
  while [[ $continue != Y && $continue != y ]]
  do
    read -a continue -p "Continue(Y/y): "
  done

  echo "Evaluating on $var..."
  sleep 5s

  # transform to TUM format
  rosrun rpg_trajectory_evaluation bag_to_pose.py \
    $WORK_DIR/Evaluate-$var.bag \
    /svo/pose_imu --out=stamped_traj_estimate.txt

  # copy groundtruth to evaluate directory
  cp $EVALUATE_OUTPUT_DIR/../stamped_groundtruth.txt $WORK_DIR

  # evaluate
  cd $WORK_DIR || exit; evo_ape tum stamped_groundtruth.txt stamped_traj_estimate.txt -vap

done