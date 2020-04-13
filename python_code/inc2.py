import sqlite3
import csv
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
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

gb = GaussianNB()
con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
cur = con.cursor()
df = pd.read_csv("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/new.csv")
df.to_sql('new', con, if_exists='append', index=False)
a = cur.execute("select * from 'new'")
realse = a.fetchall()
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
features_train, features_test, target_train, target_test = train_test_split(features, target1, test_size=0.33,                                                                                random_state=10)
gb.fit(features_train, target_train)
print("done")
joblib.dump(gb, 'model.pkl')