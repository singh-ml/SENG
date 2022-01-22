#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1
#$ -ac d=nvcr-pytorch-2112
#$ -v PATH=/home/dinesh/anaconda3/bin:$PATH

bash nvcr-pytorch-2112.sh

for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			for m in 10 20 24
			do
				python main_lbfgs.py --epoch $me --arch 'vgg16' --lr-decay-epoch $me -m $m --trainset 'cifar100' --datadir /data/ghighdim/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 0 >raiden-lbfgs_cifar100-vgg16-lr$lr-me$me-wd$wd-m$m.hist
			done
		done
	done
done

