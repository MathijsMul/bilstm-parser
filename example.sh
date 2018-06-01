TRAIN_FILE='data/train-stanford-raw.conll'
TEST_FILE='data/test-stanford-raw.conll'

RUN=1
NUM_EPOCHS=3
MODEL_NAME='myfirstmodel'
WORD_DIM=50
POS_DIM=25
NUM_LAYERS_LSTM=2
HIDDEN_UNITS_LSTM=128
HIDDEN_UNITS_MLP=64
L2=1e-4

python3 main.py --train $TRAIN_FILE --test $TEST_FILE --run $RUN --epochs $NUM_EPOCHS --word_dim $WORD_DIM --pos_dim $POS_DIM --num_layers_lstm $NUM_LAYERS_LSTM --hidden_units_lstm $HIDDEN_UNITS_LSTM --hidden_units_mlp $HIDDEN_UNITS_MLP --l2 $L2 --model_name $MODEL_NAME
