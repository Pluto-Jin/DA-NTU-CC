from shutil import copyfile
import os

os.chdir(r'./../dataset/CrowdCounting')

a = 'Train Test Splitting list/normal_training/NTU_train_correct.txt'
b = 'new_split_list/train.txt'

def copy(f,des):
    folder,name = [],[]
    with open(f) as f:
        lines = f.read().splitlines()
        for line in lines:
            tmp = line.split(' ')
            if len(tmp) == 1:
                tmp = ['hall'] + tmp
            folder.append(tmp[0])
            name.append(tmp[1].split('.')[0])

    for i in range(len(folder)):
        path = os.path.join(folder[i],'pngs_544_960',name[i]+'.png')
        copyfile(path,des+name[i]+'.png')

copy(a,'trainA/')
copy(b,'trainB/')


