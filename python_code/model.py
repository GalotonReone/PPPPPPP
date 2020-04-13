#from random import seed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame
import sqlite3
from pandas.plotting import scatter_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
import math

from sklearn.impute import SimpleImputer
from sklearn.svm.libsvm import predict_proba
from sklearn.decomposition import PCA
plt.interactive(False)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model, metrics, tree, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
cur = con.cursor()
a = cur.execute("select * from 'hist'")
realse = a.fetchall()
placement=pd.read_csv('hist.csv')
print(placement.head())
data = pd.DataFrame(np.array(realse))
print(data)
le=LabelEncoder()
name=['hsc','ssc','verbal','comm','quantitative','logical','cert','extracurr','companies','recruited']
placement=pd.read_csv('hist.csv',names=name)
placement=placement.values
for i in range(0,10):
    placement[:,i]=le.fit_transform(placement[:,i])
X=placement[:,0:9]
Y=placement[:,9]
Y=Y.astype('int')
seed=7

#data1=pd.read_csv("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/hist.csv",name=name)
dict1={'Yes':1,'No':0,'Others':0,'Cultural':1,'Sports':2,'Excellent':1,'Good':2,'Average':0,'Poor':3}
target = data.iloc[:, 18]
va = data.iloc[:, 13]
qu = data.iloc[:, 12]
lr = data.iloc[:, 11]
co = data.iloc[:, 10]
ec = data.iloc[:, 9]
dr = data.iloc[:, 19]
le = LabelEncoder()
target1 = le.fit_transform(target.astype(str))
va1 = le.fit_transform(va.astype(str))
qu1 = le.fit_transform(qu.astype(str))
lr1 = le.fit_transform(lr.astype(str))
co1 = le.fit_transform(co.astype(str))
ec1 = le.fit_transform(ec.astype(str))
dr1 = le.fit_transform(dr.astype(str))
features = list(zip(va1, qu1, lr1, co1, ec1, dr1))
n=["verbal","quants","logical","communications","extras","drop"]
features_train, features_test, target_train, target_test = train_test_split(features, target1, test_size=0.33,                                                                                random_state=10)
print("features train",features_train)
print("features test",features_test)
print("target train",target_train)
gb = GaussianNB()
dt = DecisionTreeClassifier()
lr=LogisticRegression()
knn=KNeighborsClassifier()
svm=SVC()
svm=CalibratedClassifierCV(svm)
svmm=svm.fit(features_train,target_train)
svmp=svm.predict(features_test)
knn.fit(features_train,target_train)
lrm=lr.fit(features_train,target_train)
lrp=lr.predict(features_test)
dt=dt.fit(features_train, target_train)
gbm=gb.fit(features_train, target_train)
gbp=gb.predict(features_test)
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(features_train,target_train)

y_pred=clf.predict(features_test)
#joblib.dump(gb, 'model.pkl')
#joblib.dump(svm, 'svmmodel.pkl')
#joblib.dump(clf, 'rfmodel.pkl')
#joblib.dump(knn, 'knnmodel.pkl')
#joblib.dump(dt, 'dtmodel.pkl')
#joblib.dump(lr, 'lrmodel.pkl')
target_pred4=svm.predict(features_test)
target_pred3=knn.predict(features_test)
target_pred2=lr.predict(features_test)
target_pred1 = dt.predict(features_test)
target_pred = gb.predict(features_test)
y_pred_prob=gb.predict_proba(features_test)[:,1]
#plt.scatter(target_train,features_train)
acc_svm=metrics.accuracy_score(target_test, target_pred4, normalize="True")
con_svm=confusion_matrix(target_test, target_pred4)
class_svm=classification_report(target_test,target_pred)
acc_knn=metrics.accuracy_score(target_test, target_pred3, normalize="True")
con_knn=confusion_matrix(target_test, target_pred3)
class_knn=classification_report(target_test,target_pred3)
acc_naive =metrics.accuracy_score(target_test, target_pred1, normalize="True")
con_naive=confusion_matrix(target_test, target_pred1)
class_naive=classification_report(target_test,target_pred1)
acc_decision= metrics.accuracy_score(target_test, target_pred, normalize="True")
con_dt=confusion_matrix(target_test, target_pred)
class_dt=classification_report(target_test,target_pred)
acc_logistic=metrics.accuracy_score(target_test,target_pred2,normalize="True")
class_logistic=classification_report(target_test,target_pred2)
con_logistic=confusion_matrix(target_test, target_pred2)
acc_rf=metrics.accuracy_score(target_test,y_pred,normalize=True)
con_rf=confusion_matrix(target_test, y_pred)
class_rf=classification_report(target_test,y_pred)
print("=========================================================================")
print("====================Support Vector Machine Results=======================")
print("====================Accuracy=============================================")
print(acc_svm)
print("====================Confusion Matrix=====================================")
print(con_svm)
print("====================Classification Report================================")
print(class_svm)
print("=========================================================================")
print("==========================Naive Bayes Results============================")
print("====================Accuracy=============================================")
print(acc_naive)
print("====================Confusion Matrix=====================================")
print(con_naive)
print("====================Classification Report================================")
print(class_naive)
print("=========================================================================")
print("====================Decision Tree Results=======================")
print("===============================Accuracy===================================")
print(acc_decision)
print("====================Confusion Matrix=====================================")
print(con_dt)
print("====================Classification Report================================")
print(class_dt)
print("=========================================================================")
print("====================Random Forest Tree Results=======================")
print("====================Accuracy=============================================")
print(acc_rf)
print("====================Confusion Matrix=====================================")
print(con_rf)
print("====================Classification Report================================")
print(class_rf)
print("=========================================================================")
print("====================Logistic Regression Results=======================")
print("====================Accuracy=============================================")
print(acc_logistic)
print("====================Confusion Matrix=====================================")
print(con_logistic)
print("====================Classification Report================================")
print(class_logistic)
print("=========================================================================")
print("====================K-Means Neigbor Results===============================")
print("====================Accuracy=============================================")
print(acc_knn)
print("====================Confusion Matrix=====================================")
print(con_knn)
print("====================Classification Report================================")
print(class_knn)

#print(confusion_matrix(target_test, target_pred))
X1=("SVM","DT","NB","KNN","RFT","LR")
Y1=(acc_svm,acc_decision,acc_naive,acc_knn,acc_rf,acc_logistic)
print("Random Forest Accuracy:",acc_rf)
print("Naive Bayes Accuracy:",acc_naive)
print("Decision Tree Accuracy:",acc_decision)
print("Logistic Regression Accuracy:",acc_logistic)
print("K-Nearest Neighbors Accuracy:",acc_knn)
print("Support vector Machine Accuracy:",acc_svm)
#plt.scatter(data[10],data[18])
print(pd.DataFrame(data[18].value_counts()))
#res=placement.groupby("recruited").backlogs.value_counts(normalize=True)
#res.unstack().plot(kind='bar')
#placement.verbal.value_counts().plot(kind='bar')
#plt.xlabel('verbal')
#plt.ylabel('count')
#plt.title('backlogs vs recruit')
#plt.show()
models = []
models.append(('LR', LogisticRegression()))
models.append(('RFT', RandomForestClassifier(n_estimators=100)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#ax.set_ylim([0.8,1.1])
#plt.show()
#plt.bar(X1,Y1,align='center',width=0.3,color=['green','blue','red','yellow','cyan','black'])
#plt.xlabel('Algorithm')
#plt.ylabel('Accuracy')
#plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
#plt.ylim((0,1))
#plt.title('Algorithm comparison')
#plt.show()
#tree.plot_tree(dt)
#plt.show()
#plt.scatter(target_test, svmp)
#plt.xlabel("True Values")
#plt.ylabel("Predictions")
#plt.show()
from yellowbrick.classifier import ClassificationReport
from sklearn.naive_bayes import GaussianNB
bayes = GaussianNB()
classes=['placed','non-placed']
visualizer=ClassificationReport(bayes,classes=classes)

visualizer.fit(features_train, target_train)
visualizer.score(features_test, target_test)
#visualizer.show()
print(features_test[0])
dfm=pd.DataFrame({'feature':features_test,'Actual':target_test.flatten(),'Predicted':target_pred.flatten()})
dfm[["verbal","quants","logical","communications","extras","drop"]]=pd.DataFrame(dfm['feature'].tolist(),index=dfm.index)
print(dfm)
dfm1=dfm.drop(columns=['feature','Actual'])
print(dfm1)
res = dfm1.groupby("Predicted").f1.value_counts(normalize=True)
res.unstack().plot(kind='bar')
df1 = dfm.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()