# 暂时没有用到
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from PlotConfusionMatrix import plotMatrix

def writeData(file):
    # 加载数据
    print("Loading raw data...")
    return pd.read_csv(file, header=None, low_memory=False, encoding='latin1')

def data_process(file):
    # 删除第一列，选择第二列作为标签
    raw_data = file.drop(columns=[0])
    print("print data labels:")
    print(raw_data[1].value_counts())

    # 将非数值型的数据转换为数值型数据
    raw_data[1], attacks = pd.factorize(raw_data[1], sort=True)

    # 对原始数据进行切片，分离出特征和标签
    features = raw_data.iloc[:, 1:]  # 切片获取所有特征列，跳过标签列
    labels = raw_data.iloc[:, 1]  # 标签列
    print("Initial number of columns in features:", features.shape[1])

    # # 删除所有非数值型列
    # features = features.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    # print("Number of columns in features after removing non-numeric columns:", features.shape[1])

    # 特征数据标准化，这一步是可选项
    features = preprocessing.scale(features)
    features = pd.DataFrame(features)

    # 将多维的标签转为一维的数组
    labels = labels.values.ravel()

    # 将数据分为训练集和测试集,并打印维数
    df = pd.DataFrame(features)
    X_train, X_test, y_train, y_test = train_test_split(df, labels, train_size=0.8, test_size=0.2, stratify=labels)
    print("X_train,y_train:", X_train.shape, y_train.shape)
    print("X_test,y_test:", X_test.shape, y_test.shape)
    return attacks, X_train, X_test, y_train, y_test

def training(X_train, y_train):
    # 训练模型
    print("Training model...")
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=12, min_samples_leaf=1, splitter="best")
    trained_model = clf.fit(X_train, y_train)
    print("Score:", trained_model.score(X_train, y_train))
    return clf

def testing(clf, X_test, y_test):
    # 预测
    print("Predicting...")
    y_pred = clf.predict(X_test)
    print("Computing performance metrics...")
    results = confusion_matrix(y_test, y_pred)
    error = zero_one_loss(y_test, y_pred)
    print("Error", error)
    total_predictions = np.sum(results)
    correct_predictions = np.trace(results)
    accuracy = correct_predictions / total_predictions
    print("决策树预测准确率: ", accuracy)
    return y_pred

if __name__ == '__main__':
    # 原始文件
    file = "table.csv"
    raw_data = writeData(file)
    attacks, X_train, X_test, y_train, y_test = data_process(raw_data)
    clf = training(X_train, y_train)
    y_pred = testing(clf, X_test, y_test)

    # 绘制混淆矩阵
    plotMatrix(attacks, y_test, y_pred)
