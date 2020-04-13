import csv
import gzip
import pickle

from flask import Flask, render_template, request, url_for, session
from flask_mail import Mail,Message
import random,copy
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from pandas import DataFrame
import sqlite3
from math import sqrt
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model,metrics,tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets, linear_model,metrics,tree
from sklearn.svm.libsvm import predict_proba
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sqlalchemy import null
msg1="null"
app=Flask(__name__)
mail=Mail(app)
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT']=465
app.config['MAIL_USERNAME']='chaughule20.cc@gmail.com'
app.config['MAIL_PASSWORD']='saibaba1918'
app.config['MAIL_USE_TLS']=False
app.config['MAIL_USE_SSL']=True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
mail=Mail(app)
app.secret_key='chinmay'
@app.route('/notify',methods=['POST'])
def notify():
    global text
    msg=Message('Placement Recommendation',sender='chaughule20.cc@gmail.com',recipients=['chaughule20.cc@gmail.com','sidshinde2309@gmail.com','kunaltemkar27@gmail.com'])
    msg.body='You are recommended for %s the company.Please do well prepare and follow thw ERP section to guide yourself.'%text
    mail.send(msg)
    return "sent"
@app.route('/back',methods=['POST'])
def back():
    return render_template('home.html')
@app.route('/logout',methods=['POST'])
def logout():
    return render_template('erplogin.html')
@app.route('/')
def my_form():
    return render_template('cal.html')
@app.route('/search')
def search():
    return render_template('search.html')
@app.route('/formupd',methods=['POST'])
def formupd():
    userid = session.get('userid', None)
    return render_template('genform.html',user=userid)
@app.route('/marks',methods=['POST'])
def marks():
    userid=session.get('userid', None)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a=cur.execute("select ssc,hsc,sem1,sem2,sem3,sem4,sem5,sem6,sem7,sem8,Name from 'details' where email=?",(userid,))
    l=[]
    l1=[]
    for i in a:
        l.append(i)
    for j in l[0]:
        l1.append(j)
    ssc=l1[0]
    hsc=l1[1]
    sem1=l1[2]
    sem2=l1[3]
    sem3=l1[4]
    sem4=l1[5]
    sem5=l1[6]
    sem6=l1[7]
    sem7=l1[8]
    sem8=l1[9]
    name=l1[10]
    return render_template('marks.html',name=name,email=userid,ssc=ssc,hsc=hsc,sem1=sem1,sem2=sem2,sem3=sem3,sem4=sem4,sem5=sem5,sem6=sem6,sem7=sem7,sem8=sem8)
@app.route('/extras',methods=['POST'])
def extras():
    return render_template('extras.html')
@app.route('/log',methods=['POST'])
def logical():
    return render_template('log.html')
@app.route('/verbal',methods=['POST'])
def verbal():
    return render_template('ver.html')
@app.route('/quantitative',methods=['POST'])
def quantitative():
    return render_template('quant.html')
@app.route('/communication',methods=['POST'])
def comm():
    return render_template('com.html')
@app.route('/com1',methods=['POST'])
def com1():
    return render_template('genform.html')
#@app.route('/back',methods=['POST'])
#def back():
#    return render_template('genform.html')

def correlationCoefficient(X, Y, n):
    sum_X = 0
    sum_Y = 0
    sum_XY = 0
    squareSum_X = 0
    squareSum_Y = 0

    i = 0
    while i < n:
        # sum of elements of array X.
        sum_X = sum_X + X[i]

        # sum of elements of array Y.
        sum_Y = sum_Y + Y[i]

        # sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i]

        # sum of square of array elements.
        squareSum_X = squareSum_X + X[i] * X[i]
        squareSum_Y = squareSum_Y + Y[i] * Y[i]
        y1=n * squareSum_Y - sum_Y * sum_Y
        y2=n * squareSum_X - sum_X * sum_X
        i = i + 1
    print("abs",abs((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y)))
    if abs((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y))==0:
        corr=(float)(n * sum_XY - sum_X * sum_Y) / (float)( sqrt(0.7))
    else:
        corr = (float)(n * sum_XY - sum_X * sum_Y) / (float)( sqrt(abs((n * squareSum_X - sum_X * sum_X) * (n * squareSum_Y - sum_Y * sum_Y))))
    return corr

@app.route('/extraupd',methods=['POST','GET'])
def extraupd():
    userid=session.get('userid',None)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    if request.method == 'POST':
        value=request.form.get('mycheckbox')
        print(value)
    b=cur.execute("Update details set extras=? where email=?",(value,userid,))
    con.commit()
    return render_template('genform.html')
@app.route('/marksupd',methods=['POST','GET'])
def marksupd():
    userid = session.get('userid', None)
    ssc=request.form['ssc']
    hsc = request.form['hsc']
    sem1 = request.form['sem1']
    sem2 = request.form['sem2']
    sem3 = request.form['sem3']
    sem4 = request.form['sem4']
    sem5 = request.form['sem5']
    sem6 = request.form['sem6']
    sem7 = request.form['sem7']
    sem8 = request.form['sem8']
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    b=cur.execute("Update details set ssc=?,hsc=?,sem1=?,sem2=?,sem3=?,sem4=?,sem5=?,sem6=?,sem7=?,sem8=? where email=?",(ssc,hsc,sem1,sem2,sem3,sem4,sem5,sem6,sem7,sem8,userid,))
    con.commit()
    if b.rowcount>0:
        print("updated")
    return render_template('genform.html')
con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
cur = con.cursor()
le = LabelEncoder()
b=cur.execute("select * from 'hist'")
ratingse=b.fetchall()
ratings = pd.DataFrame(np.array(ratingse))

ssc = ratings.iloc[:, 4]
ssc1 = le.fit_transform(ssc.astype(str))
hsc = ratings.iloc[:, 5]
hsc1 = le.fit_transform(hsc.astype(str))
noc = ratings.iloc[:, 8]
noc1 = le.fit_transform(noc.astype(str))
val2 = ratings.iloc[:, 9] #extra curr
val3 = ratings.iloc[:, 10]#communication
val4 = ratings.iloc[:, 11]#logical
val5 = ratings.iloc[:, 12]#quants
val6 = ratings.iloc[:, 13]#verbal
val8 = ratings.iloc[:, 15]#certis
bl = ratings.iloc[:, 16]#backlogs
bl1 = le.fit_transform(bl.astype(str))
cgpa = ratings.iloc[:, 14]
cgpa1 = le.fit_transform(cgpa.astype(str))
val21 = le.fit_transform(val2.astype(str))
val31 = le.fit_transform(val3.astype(str))
val41 = le.fit_transform(val4.astype(str))
val51 = le.fit_transform(val5.astype(str))
val61 = le.fit_transform(val6.astype(str))
val81 = le.fit_transform(val8.astype(str))
features = list(zip(ssc1, hsc1, cgpa1, noc1, val21, val31, val41, val51, val61, val81, bl1))
con.commit()
@app.route('/predict4',methods=['POST'])
def predict4():
    userid = session.get('userid', None)
    l=[]
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    le = LabelEncoder()
    b1=cur.execute("select companies from 'hist'").fetchall()
    a = cur.execute("select * from 'details'")
    realse = a.fetchall()
    for i in realse:
        if i[1]==userid:
            keys=realse.index(i)
        else:
            pass
    reals = pd.DataFrame(np.array(realse))
    id = reals.iloc[:, 17]
    id1 = le.fit_transform(id.astype(str))
    ssc2 = reals.iloc[:, 9]
    ssc3 = le.fit_transform(ssc2.astype(str))
    hsc2 = reals.iloc[:, 8]
    hsc3 = le.fit_transform(hsc2.astype(str))
    cgpa2 = reals.iloc[:, 4]
    cgpa3 = le.fit_transform(cgpa2.astype(str))
    noc2 = reals.iloc[:, 10]
    noc3 = le.fit_transform(noc2.astype(str))
    verbal = reals.iloc[:, 11]
    logical = reals.iloc[:, 14]
    quant = reals.iloc[:, 13]
    comm = reals.iloc[:, 12]
    extra = reals.iloc[:, 16]
    dr = reals.iloc[:, 7]
    cert = reals.iloc[:, 15]
    fat = pd.DataFrame(np.array([ssc2, hsc2, cgpa2, noc2, extra, comm, quant, verbal, cert, dr]))
    dr1 = le.fit_transform(dr.astype(str))
    val121 = le.fit_transform(verbal.astype(str))
    val221 = le.fit_transform(logical.astype(str))
    val321 = le.fit_transform(quant.astype(str))
    val421 = le.fit_transform(comm.astype(str))
    val521 = le.fit_transform(extra.astype(str))
    val621 = le.fit_transform(cert.astype(str))
    features1 = list(zip(ssc3, hsc3, cgpa3, noc3, val521, val421, val221, val321, val121, val621, dr1))
    con.commit()
    global msg1
    if(msg1=='Y'):
        for i, k in enumerate(features):
            x = correlationCoefficient(k, features1[keys], len(k))
            print(x)
            if (x > 0.7 and k not in l):
                l.append(i)
        l1 = []
        for key in l:
            if b1[key] not in l1:
                l1.append(b1[key])
            print(l1)
            if ('FALSE') in l1:
                l1.remove(('FALSE'))
            df1 = DataFrame(l1, columns=['companies'])
            total = len(l1)
        return render_template("index1.html",total=total, column_names=df1.columns.values,link_column="student_id", row_data=list(df1.values.tolist()), zip=zip)
    else:
        return render_template("index1.html",msg='no comp')
text=""
@app.route('/company', methods=['POST'])
def my_form_post():
    global text
    text = request.form['text']
    le = LabelEncoder()
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    b = cur.execute("SELECT * from hist where companies=?", (text,))
    ratingse = b.fetchall()
    #print(ratingse)
    if len(ratingse)>0:
        ratings = pd.DataFrame(np.array(ratingse))
        #print(ratings)
        print("This is exec")
    else:
        b = cur.execute("SELECT * from hist where companies = 'LTI'")
        ratingse1 =b.fetchall()
        ratings=pd.DataFrame(np.array(ratingse1))
        #print(ratings)
        print("else is exec")
    b1=cur.execute("select id,Name from details").fetchall()
    a = cur.execute("select * from 'details'")
    realse = a.fetchall()
    reals = pd.DataFrame(np.array(realse))
    id = reals.iloc[:, 17]
    id1 = le.fit_transform(id.astype(str))
    ssc2 = reals.iloc[:, 9]
    ssc3 = le.fit_transform(ssc2.astype(str))
    hsc2 = reals.iloc[:, 8]
    hsc3 = le.fit_transform(hsc2.astype(str))
    cgpa2 = reals.iloc[:, 4]
    cgpa3 = le.fit_transform(cgpa2.astype(str))
    noc2 = reals.iloc[:, 10]
    noc3 = le.fit_transform(noc2.astype(str))
    verbal = reals.iloc[:, 11]
    logical= reals.iloc[:, 14]
    quant = reals.iloc[:, 13]
    comm = reals.iloc[:, 12]
    extra = reals.iloc[:, 16]
    dr = reals.iloc[:, 7]
    cert = reals.iloc[:, 15]
    fat = pd.DataFrame(np.array([ssc2, hsc2, cgpa2, noc2, extra, comm, quant, verbal, cert, dr]))
    dr1 = le.fit_transform(dr.astype(str))
    val121 = le.fit_transform(verbal.astype(str))
    val221 = le.fit_transform(logical.astype(str))
    val321 = le.fit_transform(quant.astype(str))
    val421 = le.fit_transform(comm.astype(str))
    val521 = le.fit_transform(extra.astype(str))
    val621 = le.fit_transform(cert.astype(str))
    features1 = list(zip(ssc3, hsc3, cgpa3, noc3, val521, val421, val221, val321, val121, val621, dr1))
    #print("*****Features1*********")
    #print("multiple tuple",features1)
    ssc = ratings.iloc[:, 4]
    #print(ssc)
    ssc1 = le.fit_transform(ssc.astype(float))
    hsc = ratings.iloc[:, 5]
    hsc1 = le.fit_transform(hsc.astype(float))
    noc = ratings.iloc[:, 8]
    noc1 = le.fit_transform(noc.astype(int))
    val2 = ratings.iloc[:, 9]
    val3 = ratings.iloc[:, 10]
    val4 = ratings.iloc[:, 11]
    val5 = ratings.iloc[:, 12]
    val6 = ratings.iloc[:, 13]
    val8 = ratings.iloc[:, 15]
    bl = ratings.iloc[:, 16]
    bl1 = le.fit_transform(bl.astype(str))
    cgpa = ratings.iloc[:, 14]
    cgpa1 = le.fit_transform(cgpa.astype(float))
    val21 = le.fit_transform(val2.astype(str))
    val31 = le.fit_transform(val3.astype(str))
    val41 = le.fit_transform(val4.astype(str))
    val51 = le.fit_transform(val5.astype(str))
    val61 = le.fit_transform(val6.astype(str))
    val81 = le.fit_transform(val8.astype(str))
    features = list(zip(ssc1, hsc1, cgpa1, noc1, val21, val31, val41, val51, val61, val81, bl1))
    #print("********Features 2********************")
    #print("single tuple",features)
    l = []
    print("features[1]",features[1])
    for i, k in enumerate(features1):
        x = correlationCoefficient(k, features[1], len(k))
        if (x > 0.7 and k not in l):
            l.append(i)
    l1 = []
    for key in l:
        l1.append(b1[key])
        df1 = DataFrame(l1,columns=['student_id', 'Name'])
        total=len(l1)
    con.commit()
    return render_template("index.html",name1=text,total=total,column_names=df1.columns.values,link_column="student_id", row_data=list(df1.values.tolist()), zip=zip)
    #subset = df1[['student_id', 'ssc', 'hsc', 'cgpa', 'noc', 'verbal', 'logical', 'quant', 'comm', 'extra','drop', 'certis', 'branch']]
    #tuples = (tuple(x) for x in subset.values)
    #return df1.content
@app.route('/login', methods=['GET','POST'])
def erplogin():
    l=[]
    l1=[]
    if request.method=="POST":
        userid=request.form["txtemail"]
        session['userid'] = userid
        pswd=request.form["txtupass"]
        with sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db") as con:
            cur = con.cursor()
        c = cur.execute("SELECT email from details where email= (?)", [userid])
        userexists = c.fetchone()
        if userexists:
            c = cur.execute("SELECT id from details where id = (?)", [pswd])
            passwcorrect = c.fetchone()
            if passwcorrect:
                c = cur.execute("SELECT Name,id,log_per,ver_per,quant_per from details where email = (?)", [userid])
                for rows in c:
                    l.append(rows)
                for x in l[0]:
                    l1.append(x)
                name=l1[0]
                uid=l1[1]
                lper=l1[2]
                per=l1[3]
                qper=l1[4]
                return render_template('home.html',vper=per,lper=lper,qper=qper,name=name,id=uid,email=userid)

    return render_template('erplogin.html')
@app.route('/predict1',methods=['POST'])
def predict1():
    userid = session.get('userid', None)
    algo=request.form.get('choose')
    print("algotihm name",algo)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'hist'")
    realse = a.fetchall()
    data = pd.DataFrame(np.array(realse))
    dict1 = {'Yes': 1, 'No': 0, 'others': 0, 'cultural': 1, 'sports': 2, 'Excellent': 1, 'Good': 2, 'Average': 0,
             'Poor': 3}
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
    features_train, features_test, target_train, target_test = train_test_split(features, target1, test_size=0.33,
                                                                                random_state=10)
    gb = GaussianNB()
    dt = DecisionTreeClassifier()
    dt.fit(features_train, target_train)
    gb.fit(features_train, target_train)
    loaded_model = joblib.load('model.pkl')
    dt_model=joblib.load('dtmodel.pkl')
    svm_model=joblib.load('svmmodel.pkl')
    knn_model=joblib.load('knnmodel.pkl')
    lr_model=joblib.load('lrmodel.pkl')
    rf_model=joblib.load('rfmodel.pkl')
    target_pred1 = dt.predict(features_test)
    target_pred = gb.predict(features_test)
    acc_naive = metrics.accuracy_score(target_test, target_pred1, normalize="True")
    acc_decision = metrics.accuracy_score(target_test, target_pred, normalize="True")
    print(target_pred)
    print(target_test)
    print(confusion_matrix(target_test, target_pred))
    print(acc_naive)
    print(acc_decision)
    n = cur.execute("SELECT * FROM details WHERE email=? ", (userid,))
    l=[]
    l1=[]
    for row in n:
        l.append(row)
        # print(l)
    for x in l[0]:
        l1.append(x)
    name=l1[0]
    emailid=l1[1]
    course=l1[2]
    dept=l1[3]
    cgpa=l1[4]
    percent=l1[5]
    passyr=l1[6]
    backlog=l1[7]
    hsc=l1[8]
    ssc=l1[9]
    noc=l1[10]
    verbal=l1[11]
    verbal1=dict1[verbal]
    comm=l1[12]
    comm1=dict1[comm]
    quant=l1[13]
    quant1=dict1[quant]
    logical=l1[14]
    print("logical ky",logical)
    logical1=dict1[logical]
    cert=l1[15]
    extras=l1[16]
    sem1=l1[20]
    sem2=l1[21]
    sem3=l1[22]
    sem4=l1[23]
    sem5=l1[24]
    sem6=l1[25]
    sem7=l1[26]
    sem8=l1[27]
    print("extra key",extras)
    extras1=dict1[extras]
    id=l1[17]
    dob=l1[18]
    drop=l1[19]
    lquiz=l1[34]
    vquiz=l1[35]
    qquiz=l1[36]
    drop1=dict1[drop]
    if drop1==0:
        drop1=1
    else:
        drop1=0
    predicted = loaded_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    a = loaded_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    if algo=='nb':
        algo="Naive Bayes"
        predicted = loaded_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
        a = loaded_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    elif algo=='dt':
        algo="Decision Tree"
        predicted = dt_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
        a = dt_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    elif algo=='svm':
        algo="Support Vector Machine"
        predicted=svm_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
        a = svm_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    elif algo=='rf':
        algo="Random Forest Tree"
        predicted = rf_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
        a = rf_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    elif algo=='knn':
        algo="K-Nearest Neighbour"
        predicted= knn_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
        a = knn_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    elif algo=='lr':
        algo="Logistic Regression"
        predicted = lr_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
        a = lr_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    #predicted = loaded_model.predict([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    print(predicted)
    print("probability")
    #a=loaded_model.predict_proba([[verbal1, quant1, logical1, comm1, extras1, drop1]])
    print(a)
    no=a[:,0]
    yes=a[:,1]
    print(no,yes)
    for i in no:
        c=int(i*100)
    for j in yes:
        d=int(j*100)
    print(predicted)
    global msg1
    if lquiz and qquiz and vquiz:
        if int(predicted)==1:
            msg=' you can be placed'
            msg1='Y'
            d1=c
            d2=100-d1
        else:
            msg= 'sorry you cant  place! you need to work hard!!'
            d1=c
            d2=100-d1
            msg1='N'
    elif sem1==None or sem2==None or sem3==None or sem4==None:
        msg="Need to complete courses upto sem4"
        d1=0
        d2=0
    else:
        msg='Quiz for Logical/Quantitaive/Verbal not done'
        d1=0
        d2=0
    return render_template('result.html',msg=msg,no=d1,yes=d2,algo=algo)

@app.route('/predict3',methods=['GET','POST'])
def predict3():
    userid = session.get('userid', None)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'details' where email=?",(userid,))
    realse = a.fetchall()
    data = pd.DataFrame(np.array(realse))
    percentd = data.iloc[:, 5]
    backlogd = data.iloc[:, 7]
    verbald = data.iloc[:, 11]
    commd = data.iloc[:, 12]
    quantdd = data.iloc[:, 13]
    logical = data.iloc[:, 14]
    verbald=verbald.values[0]
    commd=commd.values[0]
    quantdd=quantdd.values[0]
    logical=logical.values[0]
    backlogd=backlogd.values[0]
    percentd=float(percentd.values[0])
    if verbald=='Average' or verbald=='Poor':
        ver="Verbal needs to improve"
    else :
        ver="verbal ok"
    if commd=='Average' or commd=='Poor':
        com="Communication needs to improve"
    else:
        com="communication ok"
    if quantdd=='Average' or quantdd=='Poor':
        qua="Quants needs to be improve"
    else :
        qua="quantitative ok"
    if logical=='Average' or logical=='Poor':
        log="Logical skills need to improve"
    else:
        log="Logical ok"
    if backlogd=='nil' or backlogd=='Nil' or backlogd=='Nill':
        bac="Dont allow the backlog in future"
    else:
        bac="Remove the backlog"
    if percentd>60:
        per="Cgpa is ok,maintain the cgpa as it is"
    else:
        per="Focus on cgpa"
    return render_template('requirement.html',ver=ver,com=com,qua=qua,log=log,bac=bac,per=per)

@app.route('/insert', methods=['GET','POST'])
def insert():
    msg="msg"
    if request.method == "POST":
        try:
            userid = request.form["userid"]
            pswd = request.form["pass"]
            with sqlite3.connect("C:/Users/Chinmay Chaughule/data.db") as con:
                cur = con.cursor()
                cur.execute("INSERT into details (userid,pswd) values (?,?)", (userid, pswd))
                con.commit()
                msg = "Employee successfully Added"
                con.closed()
        except:
            con.rollback()
            msg = "We can not add the employee to the list"
    return render_template('valid.html', msg=msg)

@app.route('/profile', methods=['GET','POST'])
def profile():
    userid = session.get('userid', None)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'details' where email=?", (userid,))
    realse = a.fetchall()
    data = pd.DataFrame(np.array(realse))
    return render_template('profile.html',data=realse)
original_questions = {
'Synonyms for CORPULENT':['Obese','Lean','Gaunt','Emaciated'],
'Synonyms for BRIEF':['Short','Limited','Small','Little'],
'Synonym for EMBEZZLE':['Misappropriate','Balance','Remunerate','Clear'],
'Synonym for VENT':['Opening','Stodge','End','Past tense of go'],
'Synonym for AUGUST':['Dignified','Common','Ridiculous','Petty'],
'Synonym for CANNY':['Clever','Obstinate','Handsome','Stout'],
'Synonym for ALERT':['Watchful','Energetic','Observant','Intelligent'],
'Fate smiles....those who untiringly grapple with stark realities f life':['on','with','over','round'],
 'The miser gazed ...... at the pile of gold coins in front of him.':['avidly','admiringly','thoughtfully','earnestly'],
'Catching the earlier train will give us the ...... to do some shopping.':['chance','luck','possibility','occasion'],
'Success in this examination depends ...... hard work alone.':['on','at','over','for'],
'If you smuggle goods into the country, they may be ...... by the customs authority.':['confiscated','possessed','punished','fined'],
 'Piyush behaves strangely at times and, therefore, nobody gets ...... with him':['along','about','through','up'],
'The ruling party will have to put its own house ...... order.':['in','on','to','into'],
'The man to who I sold my house was a cheat.':['to whom I sold','to whom I sell','to who I sell','who was I sold'],
'They were all shocked at his failure in the competition.':['No correction required','were shocked at all','had all shocked at','had all shocked by'],
'He is too important for tolerating any delay.':['to tolerate','to tolerating','at tolerating','with tolerating'],
"To keeps one's temper":['To be in good mood','To become hungry','To preserve ones energy','None of these'],
'To have an axe to grind':['A private end to serve','To fail to arouse interest','To have no result','To work for both sides'],
'To drive home':['To emphasise',"To find one's roots",'To return to place of rest','Back to original position']
}
questions = copy.deepcopy(original_questions)
#print(questions)
def shuffle(q):
    selected_keys = []
    i = 0
    while i < len(q):
        current_selection = random.choice(list(q.keys()))
        #print(current_selection)
        if current_selection not in selected_keys:
            selected_keys.append(current_selection)
            i = i+1
    return selected_keys
@app.route('/verbal1',methods=['POST'])
def quiz():
    questions_shuffled = shuffle(questions)
    print(questions_shuffled)
    for i in questions.keys():
        random.shuffle(questions[i])
    return render_template('vscreen.html', q=questions_shuffled, o=questions)
per=0
@app.route('/quiz', methods=['POST'])
def quiz_answers():
    correct = 0
    for i in questions.keys():
        answered = request.form.get(i)
        print(answered)
        if original_questions[i][0] == answered:
            correct = correct+1
    userid = session.get('userid', None)
    print(correct)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    print("username is ", userid)
    cur.execute("Update details set vquiz=? where email=?", (str(correct), userid,))
    print("updated")
    con.commit()
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    b = cur.execute("select vquiz,file1,file2,file3 from details where email=?", (userid,))
    realse = b.fetchall()
    data = pd.DataFrame(np.array(realse))
    mrk = int(data.iloc[:, 0])
    global per
    per= (mrk / 15) * 100
    cur.execute("Update details set ver_per=? where email=?",(per,userid,))
    print(per)
    if per > 75:
        cur.execute("Update details set verbal=? where email=?", ("Excellent", userid,))
    elif per > 60:
        cur.execute("Update details set verbal=? where email=?", ("Good", userid,))
    elif per > 45:
        cur.execute("Update details set verbal=? where email=?", ("Average", userid,))
    else:
        cur.execute("Update details set verbal=? where email=?", ("Poor", userid,))
    con.commit()

    return render_template('genform.html',user=userid)#'<h1>Correct Answers: <u>'+str(correct)+'</u></h1>'
original_questions1 = {
'What was the day on 15th august 1947 ?' :
['Friday', 'Saturday' ,'Sunday' ,'Thursday'],


'Today is Monday. After 61 days, it will be' :
['Saturday',  'Tuesday', 'Monday', 'Sunday'],


'In an election between two candidates, one got 55% of the total valid votes, 20% of the votes were invalid. If the total number of votes was 7500, the number of valid votes that the other candidate got, was' :
['2700', '2900', '2500', '3100'],


'A bag contains 50 P, 25 P and 10 P coins in the ratio 5: 9: 4, amounting to Rs. 206. Find the number of coins of each type respectively.' :
['200,360,160', '360,160,200', '160,360,200', '200,160,300'],


'If each side of a square is increased by 25%, find the percentage change in its area?' :
['56.25', '65', '65.34', '56', '58'],


'A problem is given to three students whose chances of solving it are 1/2, 1/3 and 1/4 respectively. What is the probability that the problem will be solved?' :
['3/4', '1/2', '1/4', '7/12'],


"A and B invest in a business in the ratio 3 : 2. If 5% of the total profit goes to charity and A's share is Rs. 855, the total profit is" :
['1500', '1000', '2000', '500'],


'A bag contains 6 white and 4 black balls .2 balls are drawn at random. Find the probability that they are of same colour' :
['7/15', '8/15', '1/9', '1/2'],


'If 20% of a = b, then b% of 20 is the same as' :
[' 4% of a', '6% of a', '8% of a', '10% of a'],


'A student multiplied a number by 3/5 instead of 5/3, What is the percentage error in the calculation ?' :
['64%', '54%', '74%', '84%'],


'A trader mixes 26 kg of rice at Rs. 20 per kg with 30 kg of rice of other variety at Rs. 36 per kg and sells the mixture at Rs. 30 per kg. His profit' :
['5%', '8%', '7%', '6%'],


'Fresh fruit contains 68% water and dry fruit contains 20% water. How much dry fruit can be obtained from 100 kg of fresh fruits ?':
['40', '50', '10', '67'],


'Fresh fruit contains 68% water and dry fruit contains 20% water. How much dry fruit can be obtained from 100 kg of fresh fruits ?' :
['125%', '150%', '110%', '160%'],


'A clock is set right at 8 a.m. The clock gains 10 minutes in 24 hours will be the true time when the clock indicates 1 p.m. on the following day?' :
[' 48 min. past 12.', '45 min past 12', '46 min past 12', '41 min past 12'],


'The last day of a century cannot be':
['Tuesday', 'monday', 'wednesday', 'friday'],


'The value of a machine depreciates at the rate of 10% every year. It was purchased 3 years ago. If its present value is Rs. 8748, its purchase price was ':
['12000', '14000', '17000', '34000'],



'Two numbers are respectively 20% and 50% more than a third number. The ratio of the two numbers is':
['4:5', '3:5', '2:5', '5:4'],


'Insert the missing number :7, 26, 63, 124, 215, 342, (....)' :
['511', '344', '232', '543'],


'Out of 7 consonants and 4 vowels, how many words of 3 consonants and 2 vowels can be formed?' :
['25200', '52000', '120', '24400'],


'It was Sunday on Jan 1, 2006. What was the day of the week Jan 1, 2010?' :
['friday', 'sunday', 'monday', 'wednesday'],


'If selling price is doubled, the profit triples. Find the profit percent ?' :
['100', '120', '200', '650'],


'Two cards are drawn at random from a pack of 52 cards.what is the probability that either both are black or both are queen?' :
['55/221', '55/190', '52/221', '19/221'],


'What was the day of the week on, 16th July, 1776?' :
['tuesday', 'monday', 'wednesday', 'friday'],


'In an examination, a student scores 4 marks for every correct answer and loses 1 mark for every wrong answer. If he attempts all 60 questions and secures 130 marks, the no of questions he attempts correctly is' :
['38', '40', '31', '32'],


'The average of runs of a cricket player of 10 innings was 32. How many runs must he make in his next innings so as to increase his average of runs by 4 ?' :
['76', '79', '74', '87'],



'A grocer has a sale of Rs 6435, Rs. 6927, Rs. 6855, Rs. 7230 and Rs. 6562 for 5 consecutive months. How much sale must he have in the sixth month so that he gets an average sale of Rs, 6500 ?' :
['4991', '5467', '6453', '5987'],


'Three number are in the ratio of 3 : 4 : 5 and their L.C.M. is 2400. Their H.C.F. is':
['40', '80', '32', '232'],


'A student has to obtain 33% of the total marks to pass. He got 125 marks and failed by 40 marks. The maximum marks are ':
['500', '600', '769', '200'],


"A man spends 35% of his income on food, 25% on children's education and 80% of the remaining on house rent. What percent of his income he is left with ?":
['6', '8', '5', '2'],


'if the price of a book is first decreased by 25% and then increased by 20%, then the net change in the price will be ' :
['10', '30', '40', '20']
}


questions1 = copy.deepcopy(original_questions1)
qper=0
@app.route('/quantitative1',methods=['POST'])
def quiz1():
    questions_shuffled1 = shuffle(questions1)
    print(questions_shuffled1)
    for i in questions1.keys():
        random.shuffle(questions1[i])
    return render_template('qscreen.html', q=questions_shuffled1, o=questions1)
@app.route('/quiz1', methods=['POST'])
def quiz_answers1():
    correct = 0
    for i in questions1.keys():
        answered = request.form.get(i)
        print(answered)
        if original_questions1[i][0] == answered:
            correct = correct+1
    userid = session.get('userid', None)
    print(correct)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    print("username is ", userid)
    cur.execute("Update details set qquiz=? where email=?", (str(correct), userid,))
    print("updated")
    con.commit()
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    b = cur.execute("select qquiz,file1,file2,file3 from details where email=?", (userid,))
    realse = b.fetchall()
    data = pd.DataFrame(np.array(realse))
    mrk = int(data.iloc[:, 0])
    global qper
    qper = (mrk / 15) * 100
    print(qper)
    cur.execute("Update details set quant_per=? where email=?",(qper,userid,))
    if qper > 75:
        cur.execute("Update details set quant=? where email=?", ("Excellent", userid,))
    elif qper > 60:
        cur.execute("Update details set quant=? where email=?", ("Good", userid,))
    elif qper > 45:
        cur.execute("Update details set quant=? where email=?", ("Average", userid,))
    else:
        cur.execute("Update details set quant=? where email=?", ("Poor", userid,))
    con.commit()

    return render_template('genform.html',user=userid)
original_questions2 = {
'Synonyms for CORPULENT':['Obese','Lean','Gaunt','Emaciated'],
'Synonyms for BRIEF':['Short','Limited','Small','Little'],
'Synonym for EMBEZZLE':['Misappropriate','Balance','Remunerate','Clear'],
'Synonym for VENT':['Opening','Stodge','End','Past tense of go'],
'Synonym for AUGUST':['Dignified','Common','Ridiculous','Petty'],
'Synonym for CANNY':['Clever','Obstinate','Handsome','Stout'],
'Synonym for ALERT':['Watchful','Energetic','Observant','Intelligent'],
'Fate smiles....those who untiringly grapple with stark realities f life':['on','with','over','round'],
 'The miser gazed ...... at the pile of gold coins in front of him.':['avidly','admiringly','thoughtfully','earnestly'],
'Catching the earlier train will give us the ...... to do some shopping.':['chance','luck','possibility','occasion'],
'Success in this examination depends ...... hard work alone.':['on','at','over','for'],
'Statement: Anger is energy, in a more proactive way and how to channelize it is in itself a skill. Assumptions: I. Anger need to be channelized. II. Only skillful people can channelize anger to energy.':[' If only assumption II is implicit.','If only assumption I is implicit.','if either I or II is implicit.','if neither I or II is implicit.']

}


questions2 = copy.deepcopy(original_questions2)
lper=0
@app.route('/logical',methods=['POST','GET'])
def quiz3():
    userid = session.get('userid', None)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    if request.method == 'POST':
        first=request.form.get("f",None)
        second=request.form.get("s",None)
        third=request.form.get("t",None)
        file1=request.form.get('myFile1')
        file2= request.form.get('myFile2')
        file3 = request.form.get('myFile3')
        print(file1,file2,file3)
        print(first,second,third)
    b = cur.execute( "Update details set logicalfirst=?,logicalsecond=?,logicalthird=?,file1=?,file2=?,file3=? where email=?",(first,second,third,file1,file2,file3, userid,))
    con.commit()
    #if(first!="none" && file1!=" " && second!="none")
    return render_template('genform.html')
@app.route('/logical1',methods=['POST'])
def quiz2():
    questions_shuffled2 = shuffle(questions2)
    print(questions_shuffled2)
    for i in questions2.keys():
        random.shuffle(questions2[i])
    return render_template('lscreen.html', q=questions_shuffled2, o=questions2)
@app.route('/quiz2', methods=['POST'])
def quiz_answers2():
    correct = 0
    for i in questions2.keys():
        answered = request.form.get(i)
        #print(answered)
        if original_questions2[i][0] == answered:
            correct = correct+1
    userid = session.get('userid', None)
    print(correct)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    print("username is ",userid)
    cur.execute("Update details set lquiz=? where email=?",(str(correct),userid,))
    print("updated")
    con.commit()
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    b=cur.execute("select lquiz,file1,file2,file3 from details where email=?",(userid,))
    realse = b.fetchall()
    data = pd.DataFrame(np.array(realse))
    mrk=int(data.iloc[:,0])
    global lper
    lper=(mrk/15)*100
    cur.execute("Update details set log_per=? where email=?",(lper,userid,))
    print(lper)
    if lper>75:
        cur.execute("Update details set logical=? where email=?",("Excellent",userid,))
    elif lper>60:
        cur.execute("Update details set logical=? where email=?", ("Good", userid,))
    elif lper>45:
        cur.execute("Update details set logical=? where email=?", ("Average", userid,))
    else:
        cur.execute("Update details set logical=? where email=?", ("Poor", userid,))
    con.commit()
    return render_template('genform.html',user=userid)#'<h1>Correct Answers: <u>'+str(correct)+'</u></h1>'
@app.route('/data_visualize', methods=['POST'])
def dv():
    userid = session.get('userid', None)
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'hist'")
    realse = a.fetchall()
    placement = pd.read_csv('hist.csv')
    b=cur.execute("select * from 'details'").fetchall()
    mydata = pd.DataFrame(np.array(b))
    branch=mydata.iloc[:,3]
    cgpa=mydata.iloc[:,4]
    yearofpass=mydata.iloc[:,6]
    hsc=mydata.iloc[:,8]
    ssc=mydata.iloc[:,9]
    noc=mydata.iloc[:,10]
    verbal=mydata.iloc[:,11]
    comm=mydata.iloc[:,12]
    quantitative=mydata.iloc[:,13]
    logical=mydata.iloc[:,14]
    certis=mydata.iloc[:,15]
    extracurr=mydata.iloc[:,16]
    drops=mydata.iloc[:,19]
    #cur.execute("Insert into hist(branch,cgpa,yearofpass,noc,ssc,hsc,verbal,comm,quantitative,logical,certis,extracurr,drops) values(?,?,?,?,?,?,?,?,?,?,?,?,?)",(branch,cgpa,yearofpass,noc,hsc,ssc,verbal,comm,quantitative,logical,certis,extracurr,drops,))
    con.commit()
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'hist'")
    realse = a.fetchall()
    placement = pd.read_csv('hist.csv')
    print(placement.head())
    data = pd.DataFrame(np.array(realse))
    print(data)
    name = ['hsc', 'ssc', 'verbal', 'comm', 'quantitative', 'logical', 'cert', 'extracurr', 'companies', 'recruited']
    # data1=pd.read_csv("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/hist.csv",name=name)
    dict1 = {'Yes': 1, 'No': 0, 'Others': 0, 'Cultural': 1, 'Sports': 2, 'Excellent': 1, 'Good': 2, 'Average': 0,
             'Poor': 3}
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
    features_train, features_test, target_train, target_test = train_test_split(features, target1, test_size=0.33,
                                                                                random_state=10)
    print("features train", features_train)
    print("features test", features_test)
    print("target train", target_train)
    gb = GaussianNB()
    dt = DecisionTreeClassifier()
    lr = LogisticRegression()
    knn = KNeighborsClassifier()
    svm = SVC()
    svm.fit(features_train, target_train)
    knn.fit(features_train, target_train)
    lr.fit(features_train, target_train)
    dt.fit(features_train, target_train)
    gb.fit(features_train, target_train)
    # joblib.dump(gb, 'model.pkl')
    target_pred4 = svm.predict(features_test)
    target_pred3 = knn.predict(features_test)
    target_pred2 = lr.predict(features_test)
    target_pred1 = dt.predict(features_test)
    target_pred = gb.predict(features_test)
    # predict_proba(features_test)
    # plt.scatter(target_train,features_train)
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(features_train, target_train)

    y_pred = clf.predict(features_test)
    dfm = pd.DataFrame({'feature': features_test, 'Actual': target_test.flatten(), 'Predicted': target_pred.flatten()})
    dfm[["verbal", "quants", "logical", "communications", "extras", "drop"]] = pd.DataFrame(dfm['feature'].tolist(),
                                                                                            index=dfm.index)
    #print(dfm)
    dfm1 = dfm.drop(columns=['feature', 'Actual'])
    decm = pd.DataFrame({'feature': features_test, 'Actual': target_test.flatten(), 'Predicted': target_pred1.flatten()})
    decm[["verbal", "quants", "logical", "communications", "extras", "drop"]] = pd.DataFrame(decm['feature'].tolist(),
                                                                                            index=decm.index)
    decm1 = decm.drop(columns=['feature', 'Actual'])
    lrm = pd.DataFrame({'feature': features_test, 'Actual': target_test.flatten(), 'Predicted': target_pred2.flatten()})
    lrm[["verbal", "quants", "logical", "communications", "extras", "drop"]] = pd.DataFrame(lrm['feature'].tolist(),
                                                                                            index=lrm.index)
    lrm1 = lrm.drop(columns=['feature', 'Actual'])
    knnm = pd.DataFrame({'feature': features_test, 'Actual': target_test.flatten(), 'Predicted': target_pred3.flatten()})
    knnm[["verbal", "quants", "logical", "communications", "extras", "drop"]] = pd.DataFrame(knnm['feature'].tolist(),
                                                                                            index=dfm.index)
    knnm1 = knnm.drop(columns=['feature', 'Actual'])
    svmm = pd.DataFrame({'feature': features_test, 'Actual': target_test.flatten(), 'Predicted': target_pred4.flatten()})
    svmm[["verbal", "quants", "logical", "communications", "extras", "drop"]] = pd.DataFrame(svmm['feature'].tolist(),
                                                                                            index=dfm.index)
    svmm1 = svmm.drop(columns=['feature', 'Actual'])
    rtm = pd.DataFrame({'feature': features_test, 'Actual': target_test.flatten(), 'Predicted': y_pred.flatten()})
    rtm[["verbal", "quants", "logical", "communications", "extras", "drop"]] = pd.DataFrame(rtm['feature'].tolist(),
                                                                                            index=dfm.index)
    rtm1 = rtm.drop(columns=['feature', 'Actual'])

    acc_rf = metrics.accuracy_score(target_test, y_pred, normalize=True)
    acc_svm = metrics.accuracy_score(target_test, target_pred4, normalize="True")
    acc_knn = metrics.accuracy_score(target_test, target_pred3, normalize="True")
    acc_naive = metrics.accuracy_score(target_test, target_pred1, normalize="True")
    acc_decision = metrics.accuracy_score(target_test, target_pred, normalize="True")
    acc_logistic = metrics.accuracy_score(target_test, target_pred2, normalize="True")
    rs_rft=metrics.recall_score(target_test,y_pred)
    ps_rft=metrics.precision_score(target_test,y_pred)
    f1_rft=metrics.f1_score(target_test,y_pred)
    rs_naive=metrics.recall_score(target_test,target_pred1)
    ps_naive=metrics.precision_score(target_test,target_pred1)
    f1_naive=metrics.f1_score(target_test,target_pred1)
    rs_dt=metrics.recall_score(target_test, target_pred)
    ps_dt=metrics.precision_score(target_test, target_pred)
    f1_dt=metrics.f1_score(target_test, target_pred)
    rs_svm=metrics.recall_score(target_test, target_pred4)
    ps_svm=metrics.precision_score(target_test, target_pred4)
    f1_svm=metrics.f1_score(target_test, target_pred4)
    rs_knn=metrics.recall_score(target_test, target_pred3)
    ps_knn=metrics.precision_score(target_test, target_pred3)
    f1_knn=metrics.f1_score(target_test, target_pred3)
    rs_l=metrics.recall_score(target_test, target_pred2)
    ps_l=metrics.precision_score(target_test, target_pred2)
    f1_l=metrics.f1_score(target_test, target_pred2)
    con_dt=confusion_matrix(target_test, target_pred)
    class_dt=classification_report(target_test,target_pred)
    print("confus",con_dt,"type",type(con_dt))
    print(class_dt,type(class_dt))
    print("Second member",class_dt[2])
    X1 = ("SVM", "DT", "NB", "KNN", "RFT", "LR")
    Y1 = (acc_svm, acc_decision, acc_naive, acc_knn, acc_rf, acc_logistic)

    # plt.show()
    plt.bar(X1, Y1, align='center', width=0.3, color=['green', 'blue', 'red', 'yellow', 'cyan', 'black'])
    plt.xlabel('Algorithm')
    plt.ylabel('Accuracy')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.ylim((0, 1))
    plt.title('Algorithm comparison')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/algo.png")
    plt.close()
    res = decm1.groupby("Predicted").verbal.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/vdt.png")
    plt.close()
    res = decm1.groupby("Predicted").communications.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/cdt.png")
    plt.close()
    res = decm1.groupby("Predicted").logical.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/ldt.png")
    plt.close()
    res = decm1.groupby("Predicted").quants.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/qdt.png")
    plt.close()
    res = decm1.groupby("Predicted").extras.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/edt.png")
    plt.close()
    res = decm1.groupby("Predicted").drop.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/idt.png")
    plt.close()
    res = dfm1.groupby("Predicted").verbal.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/vnb.png")
    plt.close()
    res = dfm1.groupby("Predicted").communications.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/cnb.png")
    plt.close()
    res = dfm1.groupby("Predicted").logical.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/lnb.png")
    plt.close()
    res = dfm1.groupby("Predicted").quants.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/qnb.png")
    plt.close()
    res = dfm1.groupby("Predicted").drop.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/inb.png")
    plt.close()
    res = dfm1.groupby("Predicted").extras.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/enb.png")
    plt.close()
    res = lrm1.groupby("Predicted").verbal.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/vlr.png")
    plt.close()
    res = lrm1.groupby("Predicted").communications.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/clr.png")
    plt.close()
    res = lrm1.groupby("Predicted").logical.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/llr.png")
    plt.close()
    res = lrm1.groupby("Predicted").quants.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/qlr.png")
    plt.close()
    res = lrm1.groupby("Predicted").extras.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/elr.png")
    plt.close()
    res = lrm1.groupby("Predicted").drop.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/ilr.png")
    plt.close()
    res = knnm1.groupby("Predicted").verbal.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/vkn.png")
    plt.close()
    res = knnm1.groupby("Predicted").communications.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/ckn.png")
    plt.close()
    res = knnm1.groupby("Predicted").logical.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/lkn.png")
    plt.close()
    res = knnm1.groupby("Predicted").quants.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/qkn.png")
    plt.close()
    res = knnm1.groupby("Predicted").extras.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/ekn.png")
    plt.close()
    res = knnm1.groupby("Predicted").drop.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/ikn.png")
    plt.close()
    res = svmm1.groupby("Predicted").verbal.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/vsvm.png")
    plt.close()
    res = svmm1.groupby("Predicted").communications.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/csvm.png")
    plt.close()
    res = svmm1.groupby("Predicted").logical.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/lsvm.png")
    plt.close()
    res = svmm1.groupby("Predicted").quants.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/qsvm.png")
    plt.close()
    res = svmm1.groupby("Predicted").extras.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/esvm.png")
    plt.close()
    res = svmm1.groupby("Predicted").drop.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/isvm.png")
    plt.close()

    res = rtm1.groupby("Predicted").verbal.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/vrt.png")
    plt.close()
    res = rtm1.groupby("Predicted").communications.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/crt.png")
    plt.close()
    res = rtm1.groupby("Predicted").logical.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/lrt.png")
    plt.close()
    res = rtm1.groupby("Predicted").quants.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/qrt.png")
    plt.close()
    res = rtm1.groupby("Predicted").extras.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/ert.png")
    plt.close()
    res = rtm1.groupby("Predicted").drop.value_counts(normalize=True)
    res.unstack().plot(kind='bar')
    plt.savefig("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/static/images/irt.png")
    plt.close()
    return render_template('dv.html',data=placement,cmdt=con_dt,class_dt=class_dt,rf=acc_rf,n=acc_naive,d=acc_decision,l=acc_logistic,k=acc_knn,s=acc_svm,rfr=rs_rft,rfp=ps_rft,rff=f1_rft,nr=rs_naive,np=ps_naive,nf=f1_naive,dr=rs_dt,dp=ps_dt,df=f1_dt,sr=rs_svm,sp=ps_svm,sf=f1_svm,kr=rs_knn,kp=ps_knn,kf=f1_knn,lr=rs_l,lp=ps_l,lf=f1_l)
@app.route('/model_incremental', methods=['POST'])
def modelincremental():
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'hist'")
    realse = a.fetchall()
    placement = pd.read_csv('hist.csv')
    return render_template('model1.html',data=placement)
@app.route('/modelinc', methods=['POST'])
def modelinc():
    return render_template('modelform.html')
@app.route('/refresh', methods=['POST'])
def refresh():
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'hist'")
    realse = a.fetchall()
    placement = pd.read_csv('hist.csv')
    return render_template('model1.html', data=placement)
@app.route('/sendto', methods=['POST'])
def sendto():
    userid = session.get('userid', None)
    recruit=request.form['yes']
    comp=request.form['comp']
    print(recruit,comp)
    print(type(recruit))
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'hist'")
    realse = a.fetchall()
    placement = pd.read_csv('hist.csv')
    b = cur.execute("select * from 'details' where email=?",(userid,))
    b1=b.fetchall()
    mydata = pd.DataFrame(np.array(b1))
    l=[]
    l1=[]
    for rows in b1:
        l.append(rows)
    print(l)
    for x in l[0]:
        l1.append(x)
    name=l1[0]
    branch = l1[ 3]
    cgpa = l1[4]
    yearofpass = l1[6]
    hsc = l1[8]
    ssc = l1[ 9]
    noc = l1[10]
    verbal =l1[11]
    comm = l1[12]
    quantitative = l1[13]
    logical =l1[14]
    certis = l1[15]
    extracurr =l1[16]
    drops = l1[19]
    print(drops)
    print(type(drops))
    check=cur.execute("select * from hist where name=?",(name,))
    num=check.fetchone()
    if num is None:
        cur.execute("Insert into hist(name,branch,cgpa,yearofpass,noc,ssc,hsc,verbal,comm,quantitative,logical,certis,extracurr,companies,recruited,drops) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",(name,branch,cgpa,yearofpass,noc,hsc,ssc,verbal,comm,quantitative,logical,certis,extracurr,comp,recruit,drops,))
        con.commit()
    else:
        pass
    con = sqlite3.connect("C:/Users/Chinmay Chaughule/PycharmProjects/proj1/predictor1.db")
    cur = con.cursor()
    a = cur.execute("select * from 'hist'")
    realse = a.fetchall()
    placement = pd.read_csv('hist.csv')
    con.commit()
    return render_template('model1.html',data=placement)
app.run(debug=True)