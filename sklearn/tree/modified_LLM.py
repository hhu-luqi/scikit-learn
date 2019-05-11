# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 20:47:21 2019

@author: luqi
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import time

##计算模型评价指标ap、te1、te2
def evaluate(prediction, test_y):
    ##计算tp,fp,tn,fn
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(test_y)):
        if test_y[i] == 0:
            if prediction[i] == 0:
                tn += 1
            else:
                fp += 1
        if test_y[i] == 1:
            if prediction[i] == 1:
                tp += 1
            else:
                fn += 1
    ##average precision
    ap = (tp + tn)/len(test_y)
    
    ##type 1 error
    te1 = fn/(tp + fn)
    te2 = fp/(tn + fp)
    return ap, te1, te2

class LLM:
    '''
    构建决策树和Logistic回归的混合模型LLM，
    并引入损失敏感度量cost_entropy
    方法包括训练fit、预测predict
    '''
    clf_ = None
    alpha = None
    ##存储有两类样本的叶子节点
    list_leaf = []
    ##存储叶子节点对应的LR模型
    list_LR = []
    #标准化
    scaler=StandardScaler()
    def __init__(self, max_depth=None, max_features=None, alpha=None):
        self.alpha = alpha
        
        self.clf_ = DecisionTreeClassifier(criterion = 'cost_entropy', 
                                           max_depth = max_depth, max_features = max_features)
        self.list_leaf = []            ##如果不置空，在循环实例化LLM时，list_leaf并不会重新初始化
        self.list_LR = []
        self.scaler = StandardScaler()
    def fit(self, x_train, y_train):
        '''
        参数
        x_train: array-like or np.array or DataFrame
        y_train: np.array
        alpha : int
        '''
        if self.alpha == None:
            self.clf_.fit(x_train, y_train)
        else:
            self.clf_.fit(x_train, y_train, self.alpha)
        leaf_id = self.clf_.apply(x_train)                   ##找到每个训练样本对应的叶子节点
        tmp_list_leaf = list(set(leaf_id))
           
        #标准化
        self.scaler.fit(x_train)
        train_std=self.scaler.transform(x_train)
        ##训练LR模型
        for i in tmp_list_leaf:
            index_samples = np.argwhere(leaf_id == i).reshape(-1)
            x_train_LR = train_std[index_samples, :]
            y_train_LR = y_train[index_samples]
            ##排除只包含一类样本的叶子节点，即其不需训练LR模型
            if np.sum(y_train_LR) == 0 or np.sum(y_train_LR) == len(y_train_LR):
                continue
            else:
                self.list_leaf.append(i)
                clr = LogisticRegression(C = 0.4, random_state = 1).fit(x_train_LR, y_train_LR)
                self.list_LR.append(clr)
        return self
    
    def predict(self, x_test, y_test):
        #标准化
        test_std = self.scaler.transform(x_test)
        #保存每个测试集样本对应的叶子节点
        test_leaf_id = self.clf_.apply(x_test)
        
        final_pred = np.zeros(y_test.shape)               ##用于保存所有预测值
        
        #保存测试样本出现过的所有叶子节点集合
        tmp_test_leaf = list(set(test_leaf_id))
        #save只包含一类样本的叶子节点（即该节点无LR模型）
        signal_class_leaf = [j for j in tmp_test_leaf if j not in self.list_leaf]   ##此处速度可以提高吗
        k=-1

        ##定位每个test样本所在的叶子节点,将同一节点的样本输入到对应LR模型中预测
        for i in self.list_leaf:
            k += 1
            index_samples = np.argwhere(test_leaf_id == i).reshape(-1)
            #判断测试样本是否有落在该节点中的
            if index_samples.shape[0] != 0:
                x_test_LR = test_std[index_samples, :]
                #预测
                pred_y=self.list_LR[k].predict(x_test_LR)
                
                ##将预测值存入final_pred
                for j in range(len(pred_y)):
                    final_pred[index_samples[j]] = pred_y[j]
    
        ##预测单类样本的叶子节点类别及计数其正确预测数量
        for i in signal_class_leaf:
            index_samples = np.argwhere(test_leaf_id == i).reshape(-1)
            x_test_leaf = test_std[index_samples, :]
            #决策树预测
            pred_y = self.clf_.predict(x_test_leaf)
            ##将预测值存入final_pred
            for j in range(len(pred_y)):
                final_pred[index_samples[j]] = pred_y[j]

        return final_pred

if __name__ == "__main__":
    ##读取数据为DataFrame格式
    all_data=pd.read_csv(r"C:\Users\luqi\Documents\Pattern recognition\Analysis of financial data\creditcardfraud\4_1shiyan.csv")
    #all_data.drop(['ID'], axis=1, inplace=True)
    data_Y=all_data['Class']
    data_X=all_data.drop(['Class'],axis=1)
    
    '''
    实验4.2的训练集和测试集（无数据平衡方法）
    '''
    #x_train=data_X
    #y_train=data_Y
    #test=pd.read_csv(r"C:\Users\luqi\Documents\Pattern recognition\Analysis of financial data\creditcardfraud\test4_2.csv")
    #x_test=test.iloc[:,:-1]
    #y_test=test.iloc[:,-1]
    
    ##分割数据集
    x_train,x_test,y_train,y_test = train_test_split(data_X, data_Y.values, test_size=0.2, random_state=1)

    start = time.time()
    ##使用cost-entropy 进行特征空间划分#criterion = 'cost_entropy', 
    #clf=DecisionTreeClassifier(max_depth=5)
    
    #alpha=0.000001       ##alpha = 0.000001时，tp=15（max_depth = 5）
    ##实例化LLM
    #parameter = {'max_depth':[3,4,5,6,8], 'max_features':[0.7,0.8,0.9,1]}
    clr = LLM(max_depth = 5, max_features= 0.9, alpha = 0.00001)
    clr.fit(x_train, y_train)
    list_LR = clr.list_LR
    pred_y = clr.predict(x_test, y_test)  

    end = time.time()
    ap, te1, te2 = evaluate(pred_y, y_test)
    
    print("running time is %f s" % (end-start))
    #pred_y=clf.predict(x_test)
    #accur=np.sum(np.equal(pred_y, y_test))/len(y_test)
    print("accur, type1 error, type2 error = %f, %f, %f" % (ap, te1, te2))
    #print("num of prediction 1 = %f" % np.sum(pred_y))