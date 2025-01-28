DATASET_PATH=$1
EXP_PATH=$2
OBJ=$3

echo python active_train.py -s $DATASET_PATH/${OBJ}/corrected -m ${EXP_PATH} --method=H_reg --seed=0 --override_idxs ${OBJ} --eval --schema=v20seq1_inplace --eval --resolution 2 --iterations 15000 --test_iterations 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 --densify_until_iter=8000 --sh_up_every=1000 --sh_degree=2
python active_train.py -s $DATASET_PATH/${OBJ}/corrected -m ${EXP_PATH} --method=H_reg --seed=0 --override_idxs ${OBJ} --eval --schema=v20seq1_inplace --eval --resolution 2 --iterations 15000 --test_iterations 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000 --densify_until_iter=8000 --sh_up_every=1000 --sh_degree=2