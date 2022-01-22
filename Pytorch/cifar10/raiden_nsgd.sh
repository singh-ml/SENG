#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtb-container_g1
#$ -ac d=nvcr-pytorch-2112
#$ -v PATH=/home/dinesh/anaconda3/bin:$PATH

bash nvcr-pytorch-2112.sh

for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			for irho in 10 5 2
			do
				python3 main_nsgd.py --batch-size=256 --epoch $me --arch 'vgg16' --lr-decay-epoch $me --bh 32 --irho $irho --trainset 'cifar10' --datadir /data/ghighdim/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 0 >\ raiden-nsgd_cifar10-vgg19-lr$lr-me$me-wd$wd-irho$irho.hist
			done
		done
	done
done

