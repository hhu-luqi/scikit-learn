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
class LLM:
    clf_ = None
    ##存储叶子节点
    list_leaf = []
    ##存储各个叶子节点对应的LR模型
    list_LR = []
    #标准化
    scaler=StandardScaler()
    def __init__(self, clf):
        self.clf_ = clf
        
    def build(self, x_train, y_train, alpha):
        '''
        参数
        x_train: array-like or np.array or DataFrame
        y_train: np.array
        alpha : int
        '''
        self.clf_.fit(x_train, y_train, alpha)
        leaf_id = self.clf_.apply(x_train)                   ##找到每个训练样本对应的叶子节点
        self.list_leaf = list(set(leaf_id))
           
        #标准化
        self.scaler.fit(x_train)
        train_std=self.scaler.transform(x_train)
        ##训练LR模型
        for i in self.list_leaf:
            index_samples = np.argwhere(leaf_id == i).reshape(-1)
            x_train_LR = train_std[index_samples, :]
            y_train_LR = y_train[index_samples]
            clr = LogisticRegression(random_state = 1).fit(x_train_LR, y_train_LR)
            self.list_LR.append(clr)
        return self
    
    def predict(self, x_test, y_test):
        #标准化
        test_std = self.scaler.transform(x_test)
        test_leaf_id = self.clf_.apply(x_test)
        k=-1
        accur_y = 0
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
            
        ##计算总预测精度
        accury = accur_y/len(y_test)
        return accury
        
if __name__ == "__main__":
    ##读取数据为DataFrame格式
    all_data=pd.read_excel(r"C:\Users\luqi\Documents\Pattern recognition\Analysis of financial data\data\default of credit card clients0.xls")
    all_data.drop(['ID'], axis=1, inplace=True)
    data_Y=all_data['default payment next month']
    data_X=all_data.drop(['default payment next month'],axis=1)
    
    ##分割数据集
    x_train,x_test,y_train,y_test = train_test_split(data_X, data_Y.values, test_size=0.25, random_state=1)
    ##使用二次判别分析进行分类
    clf=DecisionTreeClassifier(criterion='cost_entropy', max_depth=3)
    
    alpha=0.1
    ##实例化LLM
    clr = LLM(clf)
    clr.build(x_train, y_train, alpha)
    list_LR = clr.list_LR
    accury = clr.predict(x_test, y_test)  
    
    
    #pred_y=clf.predict(x_test)
    #accur=np.sum(np.equal(pred_y, y_test))/len(y_test)
    print("accury = %f" % accury)
    #print("num of prediction 1 = %f" % np.sum(pred_y))
