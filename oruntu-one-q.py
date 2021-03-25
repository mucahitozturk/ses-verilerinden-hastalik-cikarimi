# -*- coding: utf-8 -*-
"""
@author: Mücahit Öztürk
         Harran Üniversitesi Bilgisayar Mühendisliği     
         Örüntü Tanıma Final Sınavı Ödevi - 1. Soru
"""
#baslangic kutuphaneleri
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#veri kumesinin kolonlarını data ve target olarak cektim
veriler = pd.read_csv('data1.csv')
data = veriler.iloc[:, 0:84] #kisi seslerinden özellik çıkarımı
target = veriler.iloc[:, 84] #saglıklı/hasta verisi

# Standart scaler işlemi uygulayarak ölçekleme yaptım.
from sklearn.preprocessing import StandardScaler
 
col=data.columns
feature=col.tolist() 
 
dt=data.loc[:,feature].values  
 
sc=StandardScaler()  
dt=sc.fit_transform(dt) 
data_x=pd.DataFrame(dt,columns=feature).head().T
 
data_x.head().T

#egitim ve test kümelerini sınav dosyasında istendiği üzere %80 e %20 şeklinde oluşturdum.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 0)

# PCA ile var olan kolonları daha az kolona düşürüyoruz yani öznitelik çıkarımını gerçekleştiricez.
from sklearn.decomposition import PCA
# Kaç kolona düşürmek istiyorsak n_components e o degeri veriyoruz.
pca = PCA(n_components = 5)

pca.fit(X_train)

transform_data = pca.transform(X_train)
transform_predict_data = pca.transform(X_test)

#KNN modülünü oluşturdum ve eğittim.
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(leaf_size=30,n_neighbors=5)
knn.fit(transform_data,y_train)
predictions=knn.predict(transform_predict_data) # tahminleme yaptım.
print("KNN CLASSIFIER \n")
 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

print("Accuracy_Score",accuracy_score(y_test,predictions.round()))
#hata matrisini oluşturdum
#Confusion matrix in true positive false positive true negative false negative değerlerini yansıttım.
cm = confusion_matrix(y_test,predictions.round()) 
print('Confusion matrix: \n\n',cm)
print('\n\n')
#Detay bilgi analizi
print('Classification report: \n',classification_report(y_test,predictions.round()))