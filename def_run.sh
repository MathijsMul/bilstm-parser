train_file='data/train-stanford-raw10000.conll'
#train_file='data/train-stanford-raw100.conll'
#train_file='data/train-stanford-raw10000.conll'
test_file='data/test-stanford-raw100.conll'

num_epochs=5
word_dim=100
pos_dim=25
num_layers_lstm=1
hidden_units_lstm=125
hidden_units_mlp=100
features='default'
model_name='train1000_test100'
l2=1e-4

python3 main.py --train $train_file --test $test_file --epochs $num_epochs --word_dim $word_dim --pos_dim $pos_dim --num_layers_lstm $num_layers_lstm --hidden_units_lstm $hidden_units_lstm --hidden_units_mlp $hidden_units_mlp --features $features --l2 $l2 --model_name $model_name
