TRAIN_FILE='data/train-stanford-raw100.conll'
TEST_FILE='data/test-stanford-raw10.conll'
RUN=1

NUM_EPOCHS=3
WORD_DIM=10
POS_DIM=25
NUM_LAYERS_LSTM=1
HIDDEN_UNITS_LSTM=10
HIDDEN_UNITS_MLP=5
MODEL_NAME='train100_test10'
L2=1e-4

python3 main.py --train $TRAIN_FILE --test $TEST_FILE --run $RUN --epochs $NUM_EPOCHS --word_dim $WORD_DIM --pos_dim $POS_DIM --num_layers_lstm $NUM_LAYERS_LSTM --hidden_units_lstm $HIDDEN_UNITS_LSTM --hidden_units_mlp $HIDDEN_UNITS_MLP --l2 $L2 --model_name $MODEL_NAME
