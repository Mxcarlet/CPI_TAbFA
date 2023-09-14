export CUDA_VISIBLE_DEVICES=0

cd ../

for model in LSTM_M
do

for seq_len in 24 36 60
#for seq_len in 60
do

for preLen in 1
do

python -u main.py \
  --module $model \
  --seq_len $seq_len \
  --pred_len $preLen \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --topk 9
done
done

for seq_len in 60
do

for preLen in 3 6 12
do

python -u main.py \
  --module $model \
  --seq_len $seq_len \
  --pred_len $preLen \
  --batch_size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --topk 9
done
done

done

