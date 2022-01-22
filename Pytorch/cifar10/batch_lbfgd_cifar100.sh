for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			for m in 10 20 24
			do
				python main_lbfgs.py --epoch $me --arch 'resnet18' --lr-decay-epoch $me -m $m --trainset 'cifar100' --datadir /data/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 3| tee lbfgs_cifar100-resnet18-lr$lr-me$me-wd$wd-m$m.hist
			done
		done
	done
done

