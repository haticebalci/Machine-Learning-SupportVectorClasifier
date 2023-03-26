'''Python içine gerekli kütüphaneler import edilir.Diğer kütüphaneler ilgili işlem yapılmadan önce aşağıda import edilecektir. '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''Kullanacağımız veri seti Iris veri setidir.Python'ın pandas kütüphanesinin read_csv methodu ile veri setini import ediyoruz.'''
data = pd.read_excel("Iris.xls")
print(data)

'''Iris veri seti toplamda 5 kolondan oluşmaktadır.Kolonlardan biri bağımlı değişken diğerleri ise bağımsız değişkenlerdir.Bağımsız değişken kolonlarda verilen 
ölçüm özelliklerine species kolonu için sınıflandırma yapacağız.Öncesinde bağımsız değişkenlerdeki nitelikler için bir x matrisi,bağımlı değişken için ise bir y vektörü 
oluşturacağız.'''

X=data.iloc[:,1:-1]
Y=data.iloc[:,4:] 

'''Bağımlı ve bağımsız değişkenlerimizi belirledikten sonra Iris veri seti 4 bölüme ayrılır.Bu bölümlerden %67'lik kısım olan X_train ve Y_train eğitim için kullanılırken
%33'lük kısım olan  X_test ve Y_test ise makineye tahmin ettirilmeye çalışılacaktır.'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.33,random_state=0)

'''Modele geçmeden önce verilerde Standard Scaler denilen ön işleme yapılır.Standart ölçeklendirme işlemi, verilerin her bir özelliğini ortalama değerinden çıkarmak ve standart sapmaya 
bölmek suretiyle gerçekleştirilir. Bu, verilerinizi merkezileştirir ve standart sapmaya göre ölçeklendirir, böylece her bir özellik ortalama değeri sıfır ve standart sapması bir olan 
bir dağılım şekline sahip olur.Bu ölçeklendirme yöntemi, özellikle makine öğrenimi algoritmaları gibi modellerin performansını artırmak için kullanılan ön işleme adımlarından biridir. 
Verilerin ölçeklendirilmesi,modelin daha doğru ve güvenilir sonuçlar üretmesine yardımcı olur ve aynı zamanda modelin eğitim sürecini de hızlandırır.'''

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

'''Makine öğrenmesi algoritmalarından biri olan Support Vector Classifier modellerinde sınıflandırma ve regresyon problemlerini çözmek için kullanılan bir makine öğrenimi algoritmasıdır.
 SVM, özellikle küçük boyutlu veri setleri için etkilidir ve yüksek boyutlu verilerde de iyi performans gösterir.
SVM, sınıflandırma problemleri için kullanıldığında, farklı sınıflara ait örnekleri birbirinden net bir şekilde ayıran bir karar sınırı oluşturmaya çalışır. Bu sınır,
 sınıflar arasındaki maksimum marjini (boşluğu) maksimize edecek şekilde belirlenir. Marjin, karar sınırı ve sınıflara en yakın örnekler arasındaki mesafeyi ifade eder.Bu sayede veriler 
 bir düzlem üzerinde sınıflandırılmış olur.Avantajları:
     Düşük hata oranı
     Yüksek doğruluk
     Yüksek miktarda veri ile çalışsa da performansın düşmemesidir.'''

from sklearn.svm import SVC
supv = SVC(kernel='poly', random_state = 0)
supv.fit(X_train, Y_train.values.ravel())
tahmin=supv.predict(X_test)

'''Confusion matrix,sınıflandırma problemlerinde kullanılan bir performans ölçümüdür. Karışıklık matrisi, gerçek sınıfı ve tahmin edilen sınıfı içeren bir tablodur. 
Bu tablo, dört farklı değere sahip olabilir: true positive (TP), false positive (FP), true negative (TN) ve false negative (FN).TP, modelin doğru bir şekilde bir sınıfı
 belirlediği durumlarda oluşurken, FP modelin yanlış bir şekilde bir sınıfı belirlediği durumlarda oluşur.TN, modelin bir sınıfı doğru bir şekilde olmadığını belirlediği 
 durumlarda, FN ise modelin bir sınıfı yanlış bir şekilde olmadığını belirlediği durumlarda oluşur.Karmaşıklık matrisi, bu dört sonucu bir matris içinde gösterir.''' 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,tahmin )
print(cm)
#Confusion Matrix:
#[[16  0  0]
#[ 0 19  0]
#[ 0  4 11]]  

''' Accuracy, doğru sınıflandırılmış örneklerin toplam sayısının tüm örneklerin toplam sayısına oranıdır.Accuracy = (TP + TN) / (TP + FP + TN + FN)
Accuracy, sınıflandırma modelinin tüm sınıfları doğru bir şekilde tahmin etme becerisini ölçer. Ancak, dengesiz sınıf dağılımları gibi durumlarda yanıltıcı olabilir 
ve diğer performans metrikleri ile birlikte kullanılması önerilir.'''

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, tahmin)
print(accuracy)
#Başarı oranı:0.92 olup 50 veriden 46 tanesi doğru tahmin edilmiştir.
    

