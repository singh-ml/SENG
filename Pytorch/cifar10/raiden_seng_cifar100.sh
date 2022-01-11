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
			for dm in 0.8 1.0 1.2
			do
				python main_seng.py --epoch $me --arch 'vgg16' --lr-decay-epoch $me --damping $dm --trainset 'cifar100' --datadir /data/ghighdim/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 0 >raiden-cifar100-vgg16-lr$lr-me$me-wd$wd-dm$dm.hist
			done
		done
	done
done

