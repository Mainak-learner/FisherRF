DATASET_PATH=$1
EXP_PATH=$2

echo python active_train.py -s $DATASET_PATH -m ${EXP_PATH} -i images_4 --eval --method=H_reg --seed=0 --schema all --iterations 50000
python active_train.py -s $DATASET_PATH -m ${EXP_PATH} -i images_4 --eval --method=H_reg --seed=0 --schema all --iterations 50000