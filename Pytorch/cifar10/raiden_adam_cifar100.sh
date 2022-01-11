#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -jc gtn-container_g1_dev
#$ -ac d=nvcr-pytorch-2112
#$ -v PATH=/home/dinesh/anaconda3/bin:$PATH

bash nvcr-pytorch-2112.sh

for lr in 1e-4 2e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			for b1 in 0.9 0.99
			do
				for b2 in 0.99 0.999
				do
					python main_adam.py --epoch $me --arch 'resnet50' --lr-decay-epoch $me --trainset 'cifar100' --datadir /data/ghighdim/singh/data/ --lr $lr --weight-decay $wd --beta1 $b1 --beta2 $b2 --gpu 0 >raiden-adam_cifar100-resnet50-lr$lr-me$me-wd$wd-b1$b1-b2$b2.hist
				done
			done
		done
	done
done

