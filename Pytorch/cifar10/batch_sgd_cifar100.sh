for lr in 1e-2
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			python main_sgd.py --epoch $me --arch 'resnet18' --lr-decay-epoch $me --trainset 'cifar100' --datadir /data/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 3| tee sgd_cifar100-resnet18-lr$lr-me$me-wd$wd.hist
		done
	done
done

