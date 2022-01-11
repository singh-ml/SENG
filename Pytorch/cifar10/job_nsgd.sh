#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtb-container_g1
#$ -ac d=nvcr-pytorch-2112
#$ -v PATH=/home/dinesh/anaconda3/bin:$PATH

bash nvcr-pytorch-2112.sh

for lr in $1
do
	for me in $4
	do
		for wd in $5
		do
			for irho in $6
			do
				python3 main_nsgd.py --epoch $me --arch $2 --lr-decay-epoch $me --bh 32 --irho $irho --trainset $3 --datadir /data/ghighdim/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 0 >raiden_results/$3/$2/nsgd/raiden-$3-$2-nsgd-lr$lr-me$me-wd$wd-irho$irho.hist
			done
		done
	done
done

