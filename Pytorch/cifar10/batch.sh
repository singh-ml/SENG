for ds in 'cifar100'
do
	for a in 'resnet50'
	do
		for lr in 1e-4 2e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1
		do
			qsub job_adam.sh $lr $a $ds
		done
	done
done
