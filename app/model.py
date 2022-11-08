import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt



df=pd.read_csv('train.csv')

cdf = df[['profile pic','nums/length username','fullname words','nums/length fullname','name==username','description length','external URL','private','#posts','#followers','#follows','fake']]

x = cdf.iloc[:, :11]
y = cdf.iloc[:, -1]

          
          
clf=LogisticRegression()
clf.fit(x,y)

SVM = svm.LinearSVC()
SVM.fit(x, y)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x, y)



print(clf.predict([[1,0.27,0,0,0,53,0,0,32,1000,955]]))
print(SVM.predict([[0,0.22,1,0,0,0,0,0,0,90,333]]))

round(clf.score(x,y), 4)
round(SVM.score(x,y), 4)
round(classifier.score(x,y), 4)



file=open('my_model.pkl','wb')
pickle.dump(classifier,file,protocol=2)