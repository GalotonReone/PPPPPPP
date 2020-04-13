from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

vectorizer = CountVectorizer()
con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
cur = con.cursor()
a = cur.execute("select * from 'hist1'")
data2 = pd.read_csv(open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/hist.csv'),sep=',')
#print(data2)
data2.loc[data2["recruited"]=='Yes',"recruited"]=1
data2.loc[data2["recruited"]=='No',"recruited"]=0
data2.loc[data2['verbal']=='Excellent','verbal']=1
data2.loc[data2['verbal']=='Good','verbal']=2
data2.loc[data2['verbal']=='Average','verbal']=0
data2.loc[data2['verbal']=='Poor','verbal']=3
data2.loc[data2['comm']=='Excellent','comm']=1
data2.loc[data2['comm']=='Good','comm']=2
data2.loc[data2['comm']=='Average','comm']=0
data2.loc[data2['comm']=='Poor','comm']=3
data2.loc[data2['logical']=='Excellent','logical']=1
data2.loc[data2['logical']=='Good','logical']=2
data2.loc[data2['logical']=='Average','logical']=0
data2.loc[data2['logical']=='Poor','logical']=3
data2.loc[data2['quantitative']=='Excellent','quantitative']=1
data2.loc[data2['quantitative']=='Good','quantitative']=2
data2.loc[data2['quantitative']=='Average','quantitative']=0
data2.loc[data2['quantitative']=='Poor','quantitative']=3
data2.loc[data2['extracurr']=='others','extracurr']=0
data2.loc[data2['extracurr']=='cultural','extracurr']=1
data2.loc[data2['extracurr']=='sports','extracurr']=2
data2.loc[data2["drops"]=='Yes',"drops"]=1
data2.loc[data2["drops"]=='No',"drops"]=0
print(data2.head(5))
data2_x=data2[['verbal','quantitative','logical','comm','extracurr','drops']]
data2_y=data2['recruited']
#print(data2_x)
#print(data2_y)
realse = a.fetchall()
#placement=pd.read_csv('hist.csv')
#print(placement.head())
data = pd.DataFrame(np.array(realse))
#print(data)
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
#print(len(features))
features_train, features_test, target_train, target_test = train_test_split(data2_x,data2_y, test_size=0.33,                                                                                random_state=10)
gb = GaussianNB()
#gb.fit(features_train, target_train)
#joblib.dump(gb, 'model2.pkl')
print("packet created")
names=['recruited','verbal','quantitative','logical','comm','extracurr','drops']
data1 = pd.read_csv(open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/extra.csv'),sep=',')
cv = CountVectorizer()
#print(len(data1.recruited=='Yes'))
X = cv.fit_transform(data1.verbal+data1.quantitative+data1.logical+data1.comm+data1.extracurr+data1.drops).toarray() #replaces my older feature set
print("this is c",X)
X1=pd.DataFrame(data=X[1:,1:],index=X[1:,0],columns=X[0,1:])
#print(X.shape)
print(X1)
cv1=CountVectorizer()
print(len(features_train))
#print(X)
print(features_train)


X=cv1.fit_transform(features_train)
print(X)
print(type(X))
X_df=pd.DataFrame({'Verbal':X[:,0],'Quant':X[:,1],'Logical':X[:,2],'Commun:':X[:,3],'ExtraCurr':X[:,4],'Drops':X[:,5]})
#a=X.toarray()
res=pd.concat([X,features_train],axis=0)
#print(len(target_train))
print(len(res))
print(res)
#print(a)
#gb.partial_fit(X,target_train)