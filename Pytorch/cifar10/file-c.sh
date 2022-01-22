ds='cifar100'
a='vgg16_bn'
m='ekfac'
#for lr in 1e-4 2e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1
for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
do
	for me in 85 90
	do
		for wd in 5e-4 2e-4 1e-4
		do
			#for mm in 10 20 24
			for dm in 0.8 1.0 1.2 
			#for irho in 10 5 2
			#for b1 in 0.9 0.99
			#do
			#for b2 in 0.99 0.999
			do
				mv raiden_results/$ds/$a/$m/raiden-ekfac-cifar100-vgg16_bn-lr$lr-me$me-wd$wd-dm$dm.hist raiden_results/$ds/$a/$m/raiden-$ds-$a-$m-lr$lr-me$me-wd$wd-dm$dm.hist
			#done
			done
		done
	done
done

