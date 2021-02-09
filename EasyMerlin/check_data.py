import numpy as np
import argparse,os


def get_args():
    parser = argparse.ArgumentParser(description="train.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--DATA_ROOT', '-data_root', type=str, default='./thchs30_250_demo/', help="data root")
    # parser.add_argument("--optimizers", '-optimizers', type=str, default='sgd', help="optimizers:SGD,Adam,Radam")
    args = parser.parse_args()
    return args

def check(DATA_ROOT,ty='acoustic'):
    #filelist=os.listdir(DATA_ROOT+'X_'+ty)
    #filelist=[x.split('.')[0] for x in filelist]
    filelist = sorted(os.listdir(DATA_ROOT + 'label_phone_align'))
    filelist = [x.replace('.lab', '') for x in filelist]

    k = 0
    for nm in filelist:
        f1=np.load(DATA_ROOT+'X_acoustic/'+nm+'.npz')
        f2 = np.load(DATA_ROOT + 'Y_' + ty + '/' + nm + '.npz')
        x=f1['x'].shape
        y=f2['y'].shape
        if x[0]!=y[0]:
            print(' ===>:',nm,x,y)
            k=k+1
        else:
            pass
            #print(nm,x,y)
    print(' x.shape !=y.shape:',k)


if __name__ == '__main__':

    args = get_args()
    print(args)
    DATA_ROOT = args.DATA_ROOT
    check(DATA_ROOT, ty='acoustic')
    #check(DATA_ROOT, ty='mel')