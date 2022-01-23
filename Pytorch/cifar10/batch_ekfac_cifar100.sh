for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			for dm in 0.8 1.0 1.2
			do
				python main_ekfac.py --epoch $me --arch 'resnet18' --lr-decay-epoch $me --damping $dm --trainset 'cifar100' --datadir /data/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 0| tee ekfac_cifar100-resnet18-lr$lr-me$me-wd$wd-dm$dm-f100.hist
			done
		done
	done
done

