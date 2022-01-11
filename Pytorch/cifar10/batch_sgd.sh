for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			python main_sgd.py --epoch $me --arch 'vgg19_bn' --lr-decay-epoch $me --trainset 'cifar10' --datadir /data/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 1| tee sgd_cifar10-vgg19_bn-lr$lr-me$me-wd$wd.hist
		done
	done
done

