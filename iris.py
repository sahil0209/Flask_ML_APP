import matplotlib.pyplot as plt
import pandas as ps
import numpy as np
import pickle
import seaborn as sns


df = sns.load_dataset('iris')
live = sns.load_dataset('iris')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
temp = ps.DataFrame(live.species.values,columns=['Species'])
temp['pile'] = label_encoder.fit_transform(temp)

sns.scatterplot(temp.Species,temp.pile)

y = label_encoder.fit_transform(live.species.values)
X = live.iloc[:,:-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# Importing some libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Loop for getting the accuracies of different machine learning classification algorithms
p = [RandomForestClassifier(n_estimators=10,criterion='entropy'),KNeighborsClassifier(),SVC(kernel='linear')]
acc=[]
confMatrix = []
for i in p:
    classfier = i
    classfier.fit(X_train,y_train)
    y_pred = classfier.predict(X_test)
    acc.append(accuracy_score(y_test,y_pred))
    confMatrix.append(confusion_matrix(y_test,y_pred))


# Accuracy Chart
algos = ['RFC','KNN','SVM']
plt.figure(figsize=(8,6))
plt.plot(algos,acc,color='blue')
plt.title('Accuracy Chart of Different Classification Algorithm')
plt.xlabel('Algorithms')
plt.ylabel('Accuracy')
plt.show()


FinalModel = SVC(kernel='linear')
FinalModel.fit(X_train,y_train)
y_pred_final = FinalModel.predict(X_test)
print(classification_report(y_test,y_pred=y_pred_final))


pickle.dump(FinalModel,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

y_pred_final = model.predict(X_test)

for i in range(10):
    print('y_test: ',y_test[i],'y_pred: ',y_pred_final[i])

