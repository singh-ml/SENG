#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2112
#$ -v PATH=/home/dinesh/anaconda3/bin:$PATH

bash nvcr-pytorch-2112.sh
for lr in $1
do
	for me in $4
	do
		for wd in $5
		do
			python main_adahessian.py --epoch $me --arch $2 --lr-decay-epoch $me --trainset $3 --datadir /data/ghighdim/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 0 >raiden_results/$3/$2/adahessian/raiden-$3-$2-sgd-lr$lr-me$me-wd$wd.hist
		done
	done
done
