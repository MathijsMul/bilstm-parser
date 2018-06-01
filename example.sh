TRAIN_FILE='data/train-stanford-raw.conll'
TEST_FILE='data/test-stanford-raw.conll'

RUN=1
NUM_EPOCHS=3
L2=1e-4
MODEL_NAME='myfirstmodel'

python3 main.py --train $TRAIN_FILE --test $TEST_FILE --run $RUN --epochs $NUM_EPOCHS --l2 $L2 --model_name $MODEL_NAME