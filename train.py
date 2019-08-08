from sklearn import svm
from sklearn import metrics
from random import shuffle

import joblib
import os
import numpy as np

data_path = './npydataset/'
trainData = []
testData = []
trainLabel = []
testLabel = []



def load_data(path):
    data = []
    print ('Loading ' + path)
    for file in os.listdir(path):
        print(file)
        f_name = file.split('.')[0][:5]
        x = np.load(path + file)
        if f_name == 'motor':
            data.append((x,'motor'))
        else:
            data.append((x,'nonmotor'))
    print('Done!')
    return data

def accuaracy(num1,num2):
    if num1/num2 > 1:
        return 1.0
    else:
        return round(num1/num2,3)

def count(arr,key):
    num1 = 0
    num2 = 0
    for i in arr:
        if i == key:
            num1 += 1
        else:
            num2 += 1
    return num1,num2

def train_model_SVMLinear(trainData, trainLabel, testData, testLabel):
    count_predict_motor = 0
    count_predict_nonmotor = 0
    count_real_motor = 0
    count_real_nonmotor = 0
    print("[+] Training model")
    clf = svm.SVC(kernel='linear')
    clf.fit(trainData, trainLabel)
    print("Done training")
    pd = clf.predict(testData)

    print("[+] Testing model - test labels predict: ")
    print(pd)
    print("Real test label: ")
    print(testLabel)
    print("-------------------------------------")
    print("Testing accuracy of model: ", metrics.accuracy_score(testLabel, pd))
    print("-------------------------------------")
    
    count_predict_motor,count_predict_nonmotor = count(pd,"motor")
    count_real_motor,count_real_nonmotor = count(testLabel,"motor")

    print("Accuaracy of motor: %s" % accuaracy(count_predict_motor,count_real_motor))
    print("Accuaracy of nonmotor: %s" % accuaracy(count_predict_nonmotor,count_real_nonmotor))
    
    joblib.dump(clf, "train_model.joblib")
    print("-------------------------------------")
    print('Model save as "train_model.joblib"')
    return clf

#Load data va test tren labels du doan
data = load_data(data_path)
index = int(len(data)*0.8)
shuffle(data)
for i in range(0,index):
    trainData.append(data[i][0])
    trainLabel.append(data[i][1])
for i in range(index,len(data)):
    testData.append(data[i][0])
    testLabel.append(data[i][1])
train_model_SVMLinear(trainData, trainLabel, testData, testLabel)
