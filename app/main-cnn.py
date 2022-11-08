import numpy as np
from flask import Flask,jsonify,render_template,request
import pickle


# from keras import models
#file=open('my_model.pkl','rb')

file=open('model_picklenn1.pkl','rb')
   
clf=pickle.load(file)

#file.close()

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        mydict=request.form
        Profile_pic=int(mydict['Profile_pic'])
        userlen=int(mydict['userlen'])
        fullnamewords=int(mydict['fullnamewords'])
        fullnamelen=int(mydict['fullnamelen'])
        uname_same=int(mydict['uname_same'])
        descplen=int(mydict['descplen'])
        eurl=int(mydict['eurl'])
        file=open('my_model.pkl','rb')
        clf=pickle.load(file)
        private=int(mydict['private'])
        npost=int(mydict['npost'])
        nfollowers=int(mydict['nfollowers'])
        nfollows=int(mydict['nfollows'])
        
        
        input_feature=[Profile_pic,userlen,fullnamewords,fullnamelen,uname_same,descplen,eurl,private,npost,nfollowers,nfollows]
        #input_feature=[100,1,45,1,1,0]
        #infprob=clf.predict([input_feature])[0][1]
        infprob=clf.predict_proba([input_feature])[0][1]
        
        
        infprob = infprob*100
        return render_template('result.html',inf=infprob)
   
    return render_template('index.html')
   
if __name__ == '__main__'  :
    app.run(debug=False) 
