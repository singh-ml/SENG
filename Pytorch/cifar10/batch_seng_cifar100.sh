for lr in 1e-2
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			for dm in 0.8 1.0 1.2
			do
				python main_seng.py --epoch $me --arch 'resnet18' --lr-decay-epoch $me --damping $dm --trainset 'cifar100' --datadir /data/singh/data/ --lr $lr --weight-decay $wd --lr-scheme 'cosine' --gpu 2| tee seng_cifar100-resnet18-lr$lr-me$me-wd$wd-dm$dm.hist
			done
		done
	done
done

