#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 13:27:01 2017
@author: mgezer
"""
## Digits veritabanı ile benzer görüntü bulma
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Veri Kümesini yükleyelim
digits = datasets.load_digits()
#display_img adında görüntü gösterecek bir fonksiyon tanımlanması
def goruntuGoster(goruntuNo):
    plt.imshow(digits.images[goruntuNo], cmap=plt.cm.gray_r, 
               interpolation='nearest')
    plt.show()
#  bazı veri kümesi elemanlarını ekranda göstereli
print("")    
goruntuGoster(0)
goruntuGoster(1)
goruntuGoster(111)
# Her bir 8x8 lik görüntünün vektör haline X değişkenine aktar
X = digits.data
# Benzerlik Analizi yapılacak gorntu olan vektör şekilendiriyoruz 
#satır vektörden sütuna çeviriyoruz
goruntu=4
Y=X[goruntu].reshape(1,-1)
# Cosine Benzerlik metriğini uyguluyoruz
coSim = cosine_similarity(Y, X)
"""
#sonucu Pandas Veri çercevresinin içine alıyoruz
ve en benzerden itibaren sıralama yapıyoruz
"""
cosf = pd.DataFrame(coSim).T
cosf.columns = ['similarity']
sirali=cosf.sort_values('similarity', ascending=False)
sirali=sirali.reset_index()
#ekrana bastırıyoruz 
#print(sirali)
#enbenzerin indis değerini alıyoruz
print(goruntu,"nonun Görüntüsü")
goruntuGoster(goruntu)
print(goruntu,"nolu görüntünün sınıfı",digits.target[goruntu])
enbenzer=int(sirali.iloc[1]['index'])
bdegeri=sirali.iloc[1]['similarity']
print("En benzer değerli",enbenzer,"nonun Görüntüsü ve benzerlik değeri",
      bdegeri)
goruntuGoster(enbenzer)
print(enbenzer,"nolu görüntünün sınıfı",digits.target[enbenzer])
