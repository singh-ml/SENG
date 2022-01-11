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
			for b1 in $6 
			do
				for b2 in $7
				do
					python main_adam.py --epoch $me --arch $2 --lr-decay-epoch $me --trainset $3 --datadir /data/ghighdim/singh/data/ --lr $lr --weight-decay $wd --beta1 $b1 --beta2 $b2 --gpu 0 >raiden_results/$3/$2/adam/raiden-$3-$2-adam-lr$lr-me$me-wd$wd-b1$b1-b2$b2.hist
				done
			done
		done
	done
done

