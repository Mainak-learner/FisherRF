DATASET_PATH=$1
EXP_PATH=$2
ORACLE_PATH=$3
PROCESS=$4
SEED=$5

echo "Running sector-based active training..."

CMD="python active_train_sector.py -s $DATASET_PATH -m $EXP_PATH --initial_train 5001 --seed $SEED --oracle_model_path $ORACLE_PATH --iterations 30000 --white_background --nbv_process $PROCESS --deepkgp --test_iterations 5000 10000 15000 20000 25000 30000 --save_iterations 7000 30000"

echo $CMD
eval $CMD