name="FSRA2_2block075"
data_dir="/data/modanqi/datasets/data/train"
test_dir="/data/modanqi/datasets/data/test"
gpu_ids=1
num_worker=4
lr=0.01
sample_num=3
block=2
batchsize=8
triplet_loss=0.3
num_epochs=120
pad=0
views=2

# python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker --views $views --lr $lr \
# --sample_num $sample_num --block $block --batchsize $batchsize --triplet_loss $triplet_loss --num_epochs $num_epochs \

cd checkpoints/$name
for((i=119;i<=$num_epochs;i+=10));
do
  for((p = 0;p<=$pad;p+=10));
  do
    for ((j = 1; j < 3; j++));
    do
        python test_server.py  --test_dir $test_dir --checkpoint net_$i.pth --mode $j --gpu_ids $gpu_ids --num_worker $num_worker --pad $pad
    done
  done
done


