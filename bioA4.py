
import pandas as pd 
import numpy as np
from sklearn.metrics import confusion_matrix


from sklearn import linear_model
data = pd.read_csv("TCGA-PANCAN32-L4.csv") 
data.head()
#print( data.head())

X = data.iloc[:,3:]
#print(X)

Y1 = pd.DataFrame(data['Sample_Type'])
#print(Y1)
Y2 = pd.DataFrame(data['Cancer_Type'])
#print(Y1)


Y3=np.ravel(Y1)
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(X, Y3, test_size = 0.3, random_state=66)

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(features_train,target_train)

print("")
print("")
print("======================================================")
print("Accuracy of LogisticRegression Training set score: ")
print(reg.score(features_train, target_train))
print("")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("")
print("Accuracy of LogisticRegression Test set score: ")
print(reg.score(features_test, target_test))
print("======================================================")
print("")
print("")
print("")
LogisticRegression_Score = reg.score(features_test, target_test) * 100


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
ridge = Ridge ()
parameters = {'alpha': [1.0]}
ridge_reg = GridSearchCV(ridge,parameters,cv=5)
ridge_reg.fit(features_train, target_train)
print("======================================================")
print("Accuracy of ridge on training set: ")
print(ridge_reg.score(features_train, target_train))
print("")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("")
print("Accuracy of ridge on test set: ")
print(ridge_reg.score(features_test, target_test))
print("======================================================")
print("")
print("")
print("")
Ridge_Score = ridge_reg.score(features_test, target_test) * 100


clf = linear_model.Lasso(alpha=0.01)
clf.fit(features_train, target_train)
print("======================================================")
print("Accuracy of Lasso on training set: ")
print(clf.score(features_test, target_test))
print("")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("")
print("Accuracy of Lasso on test set: ")
print(clf.score(features_test, target_test))
print("======================================================")
print("")
print("")
print("")
Lasso_Score = clf.score(features_test, target_test) * 100



from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(features_train, target_train)
print("======================================================")
print("Accuracy of RandomForestClassifier on training set: ")
print(rf.score(features_train, target_train))
print("")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("")
print("Accuracy of RandomForestClassifier on test set: ")
print(rf.score(features_test, target_test))
print("======================================================")
RandomForestClassifier_Score = rf.score(features_test, target_test) * 100


import matplotlib.pyplot as plt


objects = ('LR', 'Ridge', 'Lasso', 'RFC')
y_pos = np.arange(len(objects))
performance = [LogisticRegression_Score, Ridge_Score, Lasso_Score, RandomForestClassifier_Score]

plt.bar(y_pos, performance, align='center', alpha=0.7)
plt.xticks(y_pos, objects)
plt.ylabel('Usage')
plt.title('Accuracy Comparision')

plt.show()
