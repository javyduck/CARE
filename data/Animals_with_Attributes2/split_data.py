import os, random, shutil
import random 
random.seed(0)
def moveFile(fileDir,tarDir):
    pathDir = os.listdir(fileDir)  
    filenumber = len(pathDir)
    picknumber = int(filenumber * ratio)  
    sample = random.sample(pathDir, picknumber) 
    for name in sample:
        shutil.move(os.path.join(fileDir, name), os.path.join(tarDir, name))
    return
 
if __name__ == '__main__':
    ori_path = 'Animals_with_Attributes2/JPEGImages'  
    test_dir = 'Animals_with_Attributes2/test'  
    ratio = 0.2  # the ratio for separating the training and test dataset.
    for firstPath in os.listdir(ori_path):
        fileDir = os.path.join(ori_path, firstPath)  
        tarDir = os.path.join(test_dir, firstPath)  
        if not os.path.exists(tarDir):  
            os.makedirs(tarDir)
        moveFile(fileDir,tarDir)
    train_dir = 'Animals_with_Attributes2/train'
    os.system("mv %s %s"% (ori_path,train_dir))