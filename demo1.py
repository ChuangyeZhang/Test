#!/usr/bin/env python2  
# -*- coding: utf-8 -*-  
""" 
Created on Tue Jun 13 10:24:50 2017 
 
@author: zyy 
"""  
from sklearn.decomposition import PCA  
from sklearn.externals import joblib
from PIL import Image
import sklearn.svm as ssv  
import numpy as np  
import glob 
import cv2  
import os  
import time  




def getData_train(filePath,label): # get the full image without cutting  
    Data = []  
    num = 0  
    for childDir in os.listdir(filePath):  
        f = os.path.join(filePath, childDir)  
        data = cv2.imread(f)  
        data = cv2.resize(data,(64,128),interpolation=cv2.INTER_CUBIC)
        fileName = np.array([[childDir]]) 
        cv2.imwrite('./train/train_pos/'+str(childDir),data)  
        data = np.reshape(data, (64 * 128,3))  
       
        data.shape = 1,3,-1         
        datalebels = zip(data, label, fileName)        
        Data.extend(datalebels)  
        num += 1  
        print("%d processing: %s" %(num,childDir))
    return Data,num  

  
def getData_test(filePath,label): # get the full image without cutting  
    Data = []  
    num = 0  
    for childDir in os.listdir(filePath):  
        f = os.path.join(filePath, childDir)  
        data = cv2.imread(f)  
        data = cv2.resize(data,(64,128),interpolation=cv2.INTER_CUBIC)
        fileName = np.array([[childDir]]) 
        cv2.imwrite('./test/test_pos/'+str(childDir),data)  
        data = np.reshape(data, (64 * 128,3)) 
       
        data.shape = 1,3,-1           
        datalebels = zip(data, label, fileName) 
        
        Data.extend(datalebels)  
        num += 1  
        print("%d processing: %s" %(num,childDir))
    return Data,num  

def getData_crop_train(filePath,label): 
    Data = []  
    num = 0  
    for childDir in os.listdir(filePath):  
        f = os.path.join(filePath, childDir)  
        image = cv2.imread(f)
        [m,n,c] = image.shape
        n1 = m/(100)
        n2 = n/(150)
        k = 0
        for i in range(n1):
            for j in range(n2):
                k = k+1
                data = image[i*64+32:(i+1)*64+32,j*128+64:(j+1)*128+64,:]
                fileName = childDir.split('.')
                filename = np.array([[fileName[0] +'_' + str(k) + '.'+fileName[1]]])
                cv2.imwrite('./train/train_neg/'+fileName[0] +'_' + str(k) + '.'+fileName[1],data)               
                data = np.reshape(data, (64 * 128,3))     
                data.shape = 1,3,-1              
                
                datalebels = zip(data, label, filename)  
                Data.extend(datalebels)  
                num += 1  
        print("%d processing: %s" %(num,childDir))
    return Data,num  


def getData_crop_test(filePath,label): 
    Data = []  
    num = 0  
    for childDir in os.listdir(filePath):  
        f = os.path.join(filePath, childDir)  
        image = cv2.imread(f)
        [m,n,c] = image.shape
        n1 = m/(200)
        n2 = n/(300)
        k = 0
        for i in range(n1):
            for j in range(n2):
                k = k+1
                data = image[i*64:(i+1)*64,j*128:(j+1)*128,:]
                fileName = childDir.split('.')
                filename = np.array([[fileName[0] +'_' + str(k) + '.'+fileName[1]]])
                cv2.imwrite('./test/test_neg/'+fileName[0] +'_' + str(k) + '.'+fileName[1],data)
                
                data = np.reshape(data, (64 * 128,3)) 
       
                data.shape = 1,3,-1  
                
                datalebels = zip(data, label, filename)  
                Data.extend(datalebels)  
                num += 1 
                 
        print("%d processing: %s" %(num,childDir))
    return Data,num 


def getFeat(Data,mode): # get and save feature valuve
    num = 0  
    for data in Data:  
        image = np.reshape(data[0], (128,64,3))
        
        # gray = rgb2gray(image) # trans image to gray
        hog = cv2.HOGDescriptor((64,128), (16,16), (8,8), (8,8), 9)
        # fd = hog(image=gray, nbins=orientations, cell_size=pixels_per_cell, signed_orientation=False, cells_per_block=cells_per_block, normalise=block_norm, visualise=visualize, flatten=True)
        # fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm, visualize, normalize)

        '''
        lbp = local_binary_pattern(image=gray, P=24, R=3)
        lbp = np.array(lbp)
        lbp = lbp.flatten()
        fd = np.concatenate((fd, lbp))      
        print(fd.shape)
        fd = np.concatenate((fd, data[1])) # add label in the end of the array
        print(fd.shape)
        '''
     
        descriptors = hog.compute(image)
        
        descriptors =  descriptors.reshape((3780,)) # in order to concatenate with label in the same dismension
        filename = data[2]     
        fd_name = filename[0].split('.')[0]+'.feat' # set feature name
        
        if mode == 'train':  
            fd_path = os.path.join('./features/train/', fd_name)  
        else:  
            fd_path = os.path.join('./features/test/', fd_name)  
        
        descriptors = np.concatenate((descriptors,data[1]))
        joblib.dump(descriptors,fd_path,compress=3) # save feature to local  
        num += 1  
        print("%d saving: %s." %(num,fd_name))




if __name__ == '__main__':  
  
  
#------------------------Data Processing--------------------------------------------------  

    t0 = time.time()  
    Pos_train_filePath = r'./INRIAPerson/train_64x128_H96/pos' 
    Neg_train_filePath = r'./INRIAPerson/train_64x128_H96/neg'  
    Pos_test_filePath = r'./INRIAPerson/test_64x128_H96/pos' 
    Neg_test_filePath = r'./INRIAPerson/test_64x128_H96/neg' 
    
    Pos_train_data,Pos_train_num = getData_train(Pos_train_filePath,np.array([[1]])) 
    getFeat(Pos_train_data,'train')  
    
    Neg_train_data,Neg_train_num = getData_crop_train(Neg_train_filePath,np.array([[0]]))  
    getFeat(Neg_train_data,'train')  
    

    Pos_test_data,Pos_test_num = getData_test(Pos_test_filePath,np.array([[1]]))
    getFeat(Pos_test_data,'test')   

    Neg_test_data,Neg_test_num = getData_crop_test(Neg_test_filePath,np.array([[0]]))  
    getFeat(Neg_test_data,'test')  
       
    t1 = time.time() 
      
    print("------------------------------------------------")  
    print("Train Positive: %d" %(Pos_train_num))  
    print("Train Negative: %d" %Neg_train_num)  
    print("Train Total: %d" %(Pos_train_num + Neg_train_num))  
    print("------------------------------------------------")  
    print("Test Positive: %d" %Pos_test_num)  
    print("Test Negative: %d" %Neg_test_num) 
    print("Test Total: %d" %(Pos_test_num+Neg_test_num))  
    print("------------------------------------------------")     
    print('The cast of time is:%f'%(t1-t0))

   
#------------------------Baseline Model--------------------------------------------------  
     
    model_path = './models/hog_svm.model'  
    train_feat_path = './features/train'  
    test_feat_path = './features/test' 
    fds = []
    labels = []
    num=0  

    # load data
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):  
        num += 1 
        data = joblib.load(feat_path)  
        fds.append(data[:-1])  
        labels.append(data[-1])
        # print("%d Dealing with %s" %(num,feat_path)) 
    
   
    fds = np.array(fds)
    labels = np.array(labels)
    
    # model train
    t0 = time.time()     
    '''
    clf = ssv.SVC(kernel='rbf')
    print "Training a SVM Classifier."
    clf.fit(fds, labels)
    joblib.dump(clf, model_path)
    '''
    clf = joblib.load(model_path)   
    t1 = time.time() 
    # print("Classifier saved to {}".format(model_path))  
    # print('The cast of time is :%f seconds' % (t1-t0))
    # print(clf.score(fds,labels))   
 
    # model test
    print " "
    print "SVM result"
    num = 0    
    total = 0 
    TP = 0
    TN = 0
    FP = 0
    FN = 0 
    t0 = time.time()  
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):  
        total += 1  
        data_test = joblib.load(feat_path)  
        data_test_feat = data_test[:-1].reshape(1,-1)  
        result = clf.predict(data_test_feat)  
        if int(result) == int(data_test[-1]) :
            num += 1 
            if  int(result) == 1:
                TP += 1
            else:
                TN += 1
        else:
            if int(result) == 0:
                FN += 1
            else:
                FP += 1
        rate = float(num)/total  
    t1 = time.time()
    
    print('The classification accuracy is %f' %rate)  
    print('The cast of time is :%f seconds' % (t1-t0)) 
    print('TP: %f' %TP)  
    print('TN: %f' %TN)  
    print('TP + TN : %f' %num)
    print('FN: %f' %FN)  
    print('FP: %f' %FP) 


#------------------------PCA Model--------------------------------------------------  

    k = 30 
    model_path = './models/svm_pca_%s.model' % k 
    train_feat_path = './features/train'  
    test_feat_path = './features/test'  
    fds = [] 
    test = []  
    test_labels = []
    labels = []   
    num_train = 0  
    num_test = 0 
    
    # load data
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):   
        num_test += 1   
        data = joblib.load(feat_path)  
         
        test.append(data[:-1])   
        test_labels.append(data[-1]) 
        # print "%d Dealing with %s" %(num_test,feat_path)  
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):   
        num_train += 1   
        data = joblib.load(feat_path)       
        fds.append(data[:-1])   
        labels.append(data[-1])   
        # print "%d Dealing with %s" %(num_train,feat_path)  
 
    fds = np.array(fds,dtype = float)    
    test = np.array(test,dtype = float)  
   
    # for k in [5,10,20,30,40,50,60,70,80,100,200,800,1000,2000,3000,3500]:
    
    pca = PCA(n_components=k)
    pca.fit(fds)
    Train_new = pca.transform(fds)
    test_new = pca.transform(test)
    
    # model train
     
    
    '''
    clf = ssv.SVC(kernel='rbf')  
    print "Training a SVM Classifier."   
    clf.fit(Train_new, labels)   
    '''
    clf = joblib.load(model_path)   
     
    # model test
    print " "
    print "PCA-SVM result"
    num = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    t0 = time.time()  
    for i in range(num_test):       
        data_test_feat = test_new[i].reshape(1,-1)      
        result = clf.predict(data_test_feat)
        if int(result) == int(test_labels[i]):  
            num += 1 
            if  int(result) == 1:
                TP += 1
            else:
                    TN += 1
        else:
            if int(result) == 0:
                FN += 1
            else:
                FP += 1		    
    t1 = time.time()   
    rate = float(num)/num_test
    # f.write('d:%f time:%f rate:%f' % (k,t1-t0,rate))
    print('The classification accuracy is %f' %rate)  
    print('The cast of time is :%f seconds' % (t1-t0)) 
    print('TP: %f' %TP)  
    print('TN: %f' %TN)  
    print('TP + TN : %f' %num)
    print('FN: %f' %FN)  
    print('FP: %f' %FP)  
    # f.write('\n\n')
