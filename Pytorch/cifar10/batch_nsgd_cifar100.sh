for lr in 1e-3 5e-3 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			for irho in 10 5 2
			do
				python main_nsgd.py --epoch $me --arch 'resnet18' --lr-decay-epoch $me --bh 32 --irho $irho --trainset 'cifar100' --datadir /data/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 1| tee nsgd_cifar100-resnet18-lr$lr-me$me-wd$wd-irho$irho.hist
			done
		done
	done
done

