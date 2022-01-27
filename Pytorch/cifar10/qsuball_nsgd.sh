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
			for irho in 1000 100 50 20 10 5 2 1
			do
				qsub job_nsgd.sh $lr vgg16_bn mnist $me $wd $irho
                                qsub job_nsgd.sh $lr vgg16 mnist $me $wd $irho
                                qsub job_nsgd.sh $lr resnet18 mnist $me $wd $irho
                                qsub job_nsgd.sh $lr resnet50 mnist $me $wd $irho
			done
		done
	done
done

