DATASET_PATH=$1
EXP_PATH=$2

echo "Running sector-based active training..."

CMD="python active_train_sector.py -s $DATASET_PATH -m $EXP_PATH --iterations 30000 --test_iterations 15000 20000 25000 30000 --save_iterations 7000 30000"

echo $CMD
eval $CMD