DATASET_PATH=$1
EXP_PATH=$2
OBJ=$3

echo python active_train.py -s $DATASET_PATH/${OBJ}/corrected -m ${EXP_PATH} --method=H_reg --seed=0 --override_idxs ${OBJ} --eval --schema=v20seq1_inplace --eval --resolution 2 --iterations 40000 --test_iterations 4000 8000 12000 16000 20000 24000 28000 32000 36000 40000 --densify_until_iter=38000 --sh_up_every=1000 --sh_degree=2
python active_train.py -s $DATASET_PATH/${OBJ}/corrected -m ${EXP_PATH} --method=H_reg --seed=0 --override_idxs ${OBJ} --eval --schema=v20seq1_inplace --eval --resolution 2 --iterations 40000 --test_iterations 4000 8000 12000 16000 20000 24000 28000 32000 36000 40000 --densify_until_iter=38000 --sh_up_every=1000 --sh_degree=2