tf=0
cf=0

count_file()
{
	t=0
	c=0
	for me in 85 90
	do
	for wd in 5e-4 2e-4 1e-4
	do
	case $3 in
		'sgd')
			for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
			do
			        t=$((t+1))
                                if test -f raiden_results/$1/$2/$3/raiden-$1-$2-$3-lr$lr-me$me-wd$wd.hist; then
                                	c=$((c+1))
				else
					qsub job_$3.sh $lr $2 $1 $me $wd
                                fi
			done
			;;
		'adam')
                        for lr in 1e-4 2e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1
                        do
				for b1 in 0.9 0.99
				do
				for b2 in 0.99 0.999
                                do
				        t=$((t+1))
                                        if test -f raiden_results/$1/$2/$3/raiden-$1-$2-$3-lr$lr-me$me-wd$wd-b1$b1-b2$b2.hist; then
                                                c=$((c+1))
                                        else
                                                qsub job_$3.sh $lr $2 $1 $me $wd $b1 $b2
                                        fi
				done
                                done
                        done
                        ;;
		'lbfgs')
                        for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
                        do
				for m in 10 20 24
                                do
				        t=$((t+1))
                                        if test -f raiden_results/$1/$2/$3/raiden-$1-$2-$3-lr$lr-me$me-wd$wd-m$m.hist; then
                                                c=$((c+1))
                                        else
                                                qsub job_$3.sh $lr $2 $1 $me $wd $m
                                        fi

                                done
                        done
                        ;;
		'nsgd')
                        for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
                        do
				for irho in 10 5 2
                                do
				        t=$((t+1))
                                        if test -f raiden_results/$1/$2/$3/raiden-$1-$2-$3-lr$lr-me$me-wd$wd-irho$irho.hist; then
                                                c=$((c+1))
					else
                                        	qsub job_$3.sh $lr $2 $1 $me $wd $irho
                                        fi
                                done
                        done
                        ;;
		*)
                        for lr in 1e-3 5e-3 1e-2 5e-2 1e-1
                        do
				for dm in 0.8 1.0 1.2
				do
					t=$((t+1))
					FILE=raiden_results/$1/$2/$3/raiden-$1-$2-$3-lr$lr-me$me-wd$wd-dm$dm.hist
					if test -f "$FILE"; then
						c=$((c+1))
                                        else
                                                qsub job_$3.sh $lr $2 $1 $me $wd $dm
					fi
				done
                        done
                        ;;
	esac
	done
	done
	echo -e "\t\t"$3-\($((c*100/t))\%\)-\($c/$t\)
        cf=$((c+cf))
	tf=$((t+tf))
	#return "$t" "$c"
}

for ds in 'mnist' #'cifar10' 'cifar100'
do
	echo $ds
	for a in 'resnet18' 'vgg16_bn' 'vgg16' 'resnet50'
	do
		echo -e "\t "$a
		for p in 'sgd' 'adam' 'kfac' 'ekfac' 'lbfgs' 'seng'
		do
			count_file $ds $a $p
			#tf=$((tf+t))
			#cf=$((tc+c))
			#echo -e "\t\t "$p '('$c'/'$t') = '$((c/t))
		done
	done
done
echo -e \($cf/$tf\) = $((cf*100/tf))\%

