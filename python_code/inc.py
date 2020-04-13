import csv

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import os.path
with open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/extra.csv','r') as ro1:
    read1=csv.reader(ro1)
    new_rows=list(read1)
if os.path.isfile('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/new.csv'):
    print ("File exist")
    with open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/new.csv', 'a') as ao:
        append = csv.writer(ao)
        append.writerows(new_rows)
        print("done if")

else:
    print ("File not exist")
    with open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/hist.csv','r') as ro:
        read=csv.reader(ro)
        rows=list(read)
    with open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/new.csv','w',newline='') as wo:
        write=csv.writer(wo)
        write.writerows(rows)
    with open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/new.csv', 'a') as ao:
        append = csv.writer(ao)
        append.writerows(new_rows)
        print("done else")

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
#print(data2.head(5))
data2_x=data2[['verbal','quantitative','logical','comm','extracurr','drops']]
data2_y=data2['recruited']
#print((data2_x).toarray())
#print((data2_y).toarray())
features_train, features_test, target_train, target_test = train_test_split(data2_x,data2_y, test_size=0.33,                                                                                random_state=10)
data1 = pd.read_csv(open('C:/Users/Chinmay Chaughule/PycharmProjects/proj1/extra.csv'),sep=',')
cv = CountVectorizer(lowercase=False)
X = cv.fit_transform(data1.verbal+data1.quantitative+data1.logical+data1.comm+data1.extracurr+data1.drops).toarray()
y=cv.fit_transform(data1.recruited).toarray()
print(len(X))
print(len(y))
''''cv1=CountVectorizer(lowercase=False)
#f_t=[str (item) for item in features_train]
#f_t=[item for item in f_t if not isinstance(item,int)]
X=cv1.fit_transform(features_train).toarray()
print(type(features_train))
print(type(target_train.to_frame()))
t_t=[str (item1) for item1 in target_train]
t_t=[item1 for item1 in t_t if not isinstance(item1,int)]
y=cv1.fit_transform(target_train.to_frame()).toarray()
'''''
print(X)
print(y)
gb = GaussianNB()
gb.partial_fit(X,y)
print("model created")