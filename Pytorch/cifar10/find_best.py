import numpy as np
import matplotlib.pyplot as plt
import sys
import os


ds = sys.argv[1]
arch = sys.argv[2]


xp = sys.argv[3]
yp = sys.argv[4]
if xp == 'epoch':
    xpi = 0
else:
    xpi = 5

sign=1
if yp == 'testloss':
    ylabel = 'Test Loss (log scale)'
    ypi = 1
    sign=-1
if yp == 'testacc':
    ylabel = 'Test Acc'
    ypi = 2
if yp == 'trainloss':
    ylabel = 'Train Loss (log scale)'
    ypi = 3
    sign=-1
if yp == 'trainacc':
    ylabel = 'Train Loss'
    ypi = 4

line_type = {
    'sgd':'k-',
    'adam':'k:',
    'lbfgs':'m',
    'kfac':'b-',
    'ekfac':'b:',
    'seng':'c:',
    'nsgd':'r-'
}

lrs = ['1e-4', '2e-4', '5e-4', '1e-3', '5e-3', '1e-2', '5e-2', '1e-1']
wds = ['1e-4', '2e-4', '5e-4']


labels = []
max_v = {}
max_f = {}
for m in ['sgd', 'adam', 'kfac', 'ekfac', 'seng', 'nsgd']:
    max_v[m] = -np.inf;
    for lr in lrs:
        for me in ['85', '90']:
            for wd in wds:
                if m == 'sgd':
                    fn = 'raiden_results/' + ds + '/' + arch + '/' + m + '/raiden-' + ds + '-' + arch + '-' + m + '-lr' + lr + '-me' + me + '-wd' + wd + '.hist'
                    if not os.path.isfile(fn):
                        continue
                    
                    num_lines = sum(1 for line in open(fn))
                    if num_lines!=93 and num_lines!=98:
                        continue
                    d = np.loadtxt(fn, skiprows=8)
                    max_ = np.max(sign*d)
                    if max_ > max_v[m]:
                        max_v[m] = max_
                        max_f[m] = fn
                elif m == 'adam':
                    for b1 in ['0.9', '0.99']:
                        for b2 in ['0.99', '0.999']:
                            fn = 'raiden_results/' + ds + '/' + arch + '/' + m + '/raiden-' + ds + '-' + arch + '-' + m + '-lr' + lr + '-me' + me + '-wd' + wd + '-b1' + b1 + '-b2' + b2 + '.hist'
                            if not os.path.isfile(fn):
                                continue
                            num_lines = sum(1 for line in open(fn))
                            if num_lines!=93 and num_lines!=98:
                                continue
                            d = np.loadtxt(fn, skiprows=8)
                            max_ = np.max(sign*d)
                            if max_ > max_v[m]:
                                max_v[m] = max_
                                max_f[m] = fn
                elif m == 'nsgd':
                    for irho in ['10', '5', '2']:
                        fn = 'raiden_results/' + ds + '/' + arch + '/' + m + '/raiden-' + ds + '-' + arch + '-' + m + '-lr' + lr + '-me' + me + '-wd' + wd + '-irho' + irho + '.hist'
                        if not os.path.isfile(fn):
                            continue
                        num_lines = sum(1 for line in open(fn))
                        if num_lines!=93 and num_lines!=98:
                            continue
                        d = np.loadtxt(fn, skiprows=8)
                        max_ = np.max(sign*d)
                        if max_ > max_v[m]:
                            max_v[m] = max_
                            max_f[m] = fn
                elif m == 'lbfgs':
                    for mm in ['10', '20', '24']:
                        fn = 'raiden_results/' + ds + '/' + arch + '/' + m + '/raiden-' + ds + '-' + arch + '-' + m + '-lr' + lr + '-me' + me + '-wd' + wd + '-m' + mm + '.hist'
                        if not os.path.isfile(fn):
                            continue
                        num_lines = sum(1 for line in open(fn))
                        if num_lines!=93 and num_lines!=98:
                            continue
                        d = np.loadtxt(fn, skiprows=8)
                        max_ = np.max(sign*d)
                        if max_ > max_v[m]:
                            max_v[m] = max_
                            max_f[m] = fn
                else:
                    for dm in ['0.8', '1.0', '1.2']:
                        fn = 'raiden_results/' + ds + '/' + arch + '/' + m + '/raiden-' + ds + '-' + arch + '-' + m + '-lr' + lr + '-me' + me + '-wd' + wd + '-dm' + dm + '.hist'
                        if not os.path.isfile(fn):
                            continue
                        num_lines = sum(1 for line in open(fn))
                        if num_lines!=93 and num_lines!=98:
                            continue
                        d = np.loadtxt(fn, skiprows=8)
                        max_ = np.max(sign*d)
                        if max_ > max_v[m]:
                            max_v[m] = max_
                            max_f[m] = fn
    if m in max_f.keys():
        print(max_f[m])

for (yp, ypi) in [('trainloss',3), ('trainacc',4), ('testloss',1), ('testacc',2)]:
    if yp == 'testloss':
        ylabel = 'Test Loss (log scale)'
    if yp == 'testacc':
        ylabel = 'Test Acc'
    if yp == 'trainloss':
        ylabel = 'Train Loss (log scale)'
    if yp == 'trainacc':
        ylabel = 'Train Acc'
    for m in ['sgd', 'adam', 'kfac', 'ekfac', 'seng', 'nsgd']:
        if m in max_f.keys():
            labels.append(m.upper())
            d = np.loadtxt(max_f[m], skiprows=8)
            if ypi==1 or ypi==3:
                plt.plot(d[:, xpi], np.log(d[:, ypi]), line_type[m])
            else:
                plt.plot(d[:, xpi], d[:, ypi], line_type[m])
    plt.legend(labels)
    plt.title(ds+' - '+arch)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.savefig(ds+'_'+arch+'_'+yp+'_'+xp+'_'+sys.argv[4]+'.eps')
    plt.clf()

