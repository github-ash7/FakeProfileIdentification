import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

df=pd.read_csv('train.csv')

df.head()
df.tail()

df.info()

df.describe()



df.isnull().sum()
df['profile pic'].value_counts()

df['fake'].value_counts()

df['nums/length username'].value_counts()

sns.countplot(df['fake'])
plt.show()


sns.countplot(df['private'])
plt.show()


# Correlation plot
plt.figure(figsize=(20, 20))
cm = df.corr()
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)
plt.show()

sns.pairplot(df)

cdf = df[['profile pic','nums/length username','fullname words','nums/length fullname','name==username','description length','external URL','private','#posts','#followers','#follows','fake']]

x = cdf.iloc[:, :11]
y = cdf.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)          
          
clf=LogisticRegression()
clf.fit(x,y)
accl = clf.score(x,y)
clf_acc = accl*100

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x, y)
nbl = nb.score(x,y)
nb_acc = nbl*100
print("Naive Bayes score: ",nb.score(x, y))

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(x, y)
RFacc = RF.score(x, y)

NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(x, y)
NNacc = NN.score(x, y)



y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))


ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('top') 
ax.xaxis.tick_top()
ax.set_ylabel('True', fontsize=20)
plt.show()


SVM = svm.LinearSVC()
SVM.fit(x, y)
acc = SVM.score(x, y)
svm_acc = acc*100

print(clf.predict([[0,0.22,1,0,0,0,0,0,0,90,333]]))
print(SVM.predict([[0,0.22,1,0,0,0,0,0,0,90,333]]))

round(clf.score(x,y), 4)
round(SVM.score(x,y), 4)

data = {'LogisticRegression':clf_acc, 'SVC':svm_acc, 'RandomForest':RFacc*100, 'NN':NNacc*100, 'NB':nb_acc}
courses = list(data.keys())
values = list(data.values())


fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color =['black', 'red', 'green', 'cyan', 'pink', 'violet', 'orange'], 
        width = 0.4)
 
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy of Algorithms")
plt.show()


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

round(clf.score(x,y), 4)
print('Recall: %.3f' % recall_score(y_test, y_pred))
print('Precision: %.3f' % precision_score(y_test, y_pred))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print('F1 Score: %.3f' % f1_score(y_test, y_pred))


file=open('my_model.pkl','wb')
pickle.dump(clf,file,protocol=2)