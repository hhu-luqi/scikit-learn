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

class LLM:
    clf_ = None
    ##存储有两类样本的叶子节点
    list_leaf = []
    ##存储叶子节点对应的LR模型
    list_LR = []
    #标准化
    scaler=StandardScaler()
    def __init__(self, clf):
        self.clf_ = clf
        
    def build(self, x_train, y_train, alpha = None):
        '''
        参数
        x_train: array-like or np.array or DataFrame
        y_train: np.array
        alpha : int
        '''
        if alpha == None:
            self.clf_.fit(x_train, y_train)
        else:
            self.clf_.fit(x_train, y_train, alpha)
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
                clr = LogisticRegression(random_state = 1).fit(x_train_LR, y_train_LR)
                self.list_LR.append(clr)
        return self
    
    def predict(self, x_test, y_test):
        #标准化
        test_std = self.scaler.transform(x_test)
        #保存每个测试集样本对应的叶子节点
        test_leaf_id = self.clf_.apply(x_test)
        #保存测试样本出现过的所有叶子节点集合
        tmp_test_leaf = list(set(test_leaf_id))
        #save只包含一类样本的叶子节点（即该节点无LR模型）
        signal_class_leaf = [j for j in tmp_test_leaf if j not in self.list_leaf]   ##此处速度可以提高吗
        k=-1
        accur_y = 0
        true_fraud = 0
        ##定位每个test样本所在的叶子节点,将同一节点的样本输入到对应LR模型中预测
        for i in self.list_leaf:
            k += 1
            index_samples = np.argwhere(test_leaf_id == i).reshape(-1)
            #判断测试样本是否有落在该节点中的
            if index_samples.shape[0] != 0:
                x_test_LR = test_std[index_samples, :]
                y_test_LR = y_test[index_samples]
                #预测
                pred_y=self.list_LR[k].predict(x_test_LR)
                accur_y += np.sum(np.equal(pred_y, y_test_LR))
                ##标签1的样本被正确检测的数量
                f_pred_y = np.argwhere(pred_y == 1).reshape(-1)
                f_y_test_LR = np.argwhere(y_test_LR == 1).reshape(-1)
                ##比较两个列表中有多少个元素相同
                for p in f_pred_y:
                    if p in f_y_test_LR:
                        true_fraud += 1
    
        ##预测单类样本的叶子节点类别及计数其正确预测数量
        accur_y_leaf = 0
        for i in signal_class_leaf:
            index_samples = np.argwhere(test_leaf_id == i).reshape(-1)
            x_test_leaf = test_std[index_samples, :]
            y_test_leaf = y_test[index_samples]
            #决策树预测
            pred_y = self.clf_.predict(x_test_leaf)
            accur_y_leaf += np.sum(np.equal(pred_y, y_test_leaf))
            ##标签1的样本被正确检测的数量
            f_pred_y = np.argwhere(pred_y == 1).reshape(-1)
            f_y_test_leaf = np.argwhere(y_test_leaf == 1).reshape(-1)
            ##比较两个列表中有多少个元素相同
            for p in f_pred_y:
                if p in f_y_test_leaf:
                    true_fraud += 1
        ##计算总预测精度
        accury = (accur_y + accur_y_leaf)/len(y_test)
        return accury, true_fraud
        
if __name__ == "__main__":
    ##读取数据为DataFrame格式
    all_data=pd.read_csv(r"C:\Users\luqi\Documents\Pattern recognition\Analysis of financial data\creditcardfraud\2wan transaction.csv")
    #all_data.drop(['ID'], axis=1, inplace=True)
    data_Y=all_data['Class']
    data_X=all_data.drop(['Class'],axis=1)
    
    ##分割数据集
    x_train,x_test,y_train,y_test = train_test_split(data_X, data_Y.values, test_size=0.25, random_state=1)
    
    start = time.time()
    ##使用cost-entropy 进行特征空间划分
    clf=DecisionTreeClassifier(criterion = 'cost_entropy', max_depth=5)
    
    alpha=0.000001       ##alpha = 0.000001时，tp=15（max_depth = 5）
    ##实例化LLM
    clr = LLM(clf)
    clr.build(x_train, y_train, alpha)
    list_LR = clr.list_LR
    accury, true_fraud = clr.predict(x_test, y_test)  
    
    end = time.time()
    print("running time is %f s" % (end-start))
    #pred_y=clf.predict(x_test)
    #accur=np.sum(np.equal(pred_y, y_test))/len(y_test)
    print("accury = %f, count of TP:%d" % (accury, true_fraud))
    #print("num of prediction 1 = %f" % np.sum(pred_y))
