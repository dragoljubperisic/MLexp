# -*- coding: utf-8 -*-
"""
Created on Nov 16  2024
author: Dragoljub Perisic
postupak koji generiše fajl za primer
1. 
https://minorplanetcenter.net/db_search/
primer sadrži objekte
upit na sajtu
NEOs (Show)
na dana 14.11.2024. 36524 objekata
sa sajta, posle upita, prebaceno u Excel
pa posle iz Excela u CSV (DOS)
file name :
Set005NEOs36524.csv
btw: isti upit na dan 16.11.2024. daje 36539 objekata
ver 0.7 -- iscrtane sve 4 kategorije drugom bojom
ver 0.8 -- sredjivanje koda
ver 0.85 -- ubacena provera preseka kategorija za asteroide .. nadjeno da atens i Apohele imaju presek
ver 0.86 -- ubacena tacna provera za sve cetiri kategorije
ver 0.87 -- pravim a vs e za sve 4 kategorije
ver 0.88 --- ugradnja granica uz ML 23.11.2024. 15:15 ... proradila prva verzija
ver 0.89 --- krece rafinacija ML granica 15:35  -- proradio KMeans
ver 0.90 --- krece Fine tuning 
ver 0.91 --- priprema za finalnu verziju - u prethodnoj verziji je sve što je ovde obrisano
ver 0.92 --- priprema za finalnu verziju - samo glavno je ovde
ver 0.94 --- proradio random forest krece testiranje tacnosti
ver 0.95 ovde secemo podatke u dve grupe
ver 0.96 proradio ML algoritam i tacnost je 0.974
ver 0.97 proradio ML algoritam i tacnost je 0.974 - ovde je sve skockano  30.11.2024. 15:42
"""


import copy


MylistOfSB = []

MylistOfAmor = []
MylistOfApollo = []
MylistOfAtens = []
MylistOfApohele = []


################ klasa za putanjske elemente malog tela

"""
NEW
1  Argument of Perihelion   -PEmaloomega   -degree
2  Ascending Node           -PEvelikoOmega -degree
3  Inclination              -PEi           -degree
4  Eccentricity             -PEe           -
5  Perihelion Distance      -PEq           -AU
6  Semimajor Axis           -PEa           -AU
7  Mean Anomaly             -PEM           -degree
8  Mean Daily Motion        -PEn           -degree/day   
9  Aphelion Distance        -PEQ           -AU
10 Period                   -PEP           -years
11 Absolute magnitude       -PEH           -mag
PEmaloomega,PEvelikoOmega,PEi,PEe,PEq,PEa,PEM,PEn,PEQ,PEP,PEH
"""


class SmallBody:
  def __init__(self, Name, PEmaloomega,PEvelikoOmega,PEi,PEe,PEq,PEa,PEM,PEn,PEQ,PEP,PEH):
      
      
    self.Name=Name
    self.PEmaloomega= PEmaloomega
    self.PEvelikoOmega=PEvelikoOmega
    self.PEi=PEi  
    self.PEe=PEe 
    self.PEq=PEq     
    self.PEa=PEa
    self.PEM=PEM
    self.PEn=PEn     
    self.PEQ=PEQ
    self.PEP=PEP
    self.PEH=PEH
    
   
  def SBPrint(self):
      print (str(self.PEa) +" "+ str(self.PEe) +" "+  str(self.PEi) +" "+  str(self.PEmaloomega) +" "+  
             str(self.PEvelikoOmega) +" "+str(self.PEM) +" "+str(self.PEn) +" "+  
             str(self.PEq) +" "+  str(self.PEQ) +" "+  str(self.PEP) +" "+  str(self.PEH))
  

      print (str(self.PEmaloomega) +" " +str(self.PEvelikoOmega) +str(self.PEi) +" "+ str(self.PEe) +" "+  str(self.PEi) +" "+  " "+  
             str(self.PEq) +" "+str(self.PEa) +" "+str(self.PEq) +" "+  
             str(self.PEa) +" "+  str(self.PEM) +" "
             +  str(self.PEP) +" "+  str(self.PEH))

    
      
  def SBPrintName(self):
      print (self.Name)
      

####mySmallBodTest = SmallBody("test2",10,0.2,4,5,44,53,0.2,3.1,3.4,5.6,10.2)
####mySmallBodTest.SBPrint()

mySmallBodyTemp = SmallBody("test3",0,0,0,0,0,0,0,0,0,0,0)



import csv

# Define a class to hold the string and 11 numbers
class DataRow:
    def __init__(self, label: str, numbers: list):
        self.label = label    # A string label
        self.numbers = numbers  # A list of 11 numbers
    
    def __repr__(self):
        return f"DataRow(label={self.label}, numbers={self.numbers})"

# Function to read CSV and load data into instances of DataRow
def load_data_from_csv(file_path):
    data_rows = []
    
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        
        # Process each line in the CSV file
        for line_num, row in enumerate(csv_reader, start=1):
            if len(row) < 12:
                print(f"Line {line_num} does not contain a label and 11 numbers: {row}")
                continue
            
            try:
                # First item is the label (string), the rest should be numbers
                label = row[0]
                numbers = [float(item) for item in row[1:]]

                mySmallBodyTemp.Name=row[0]
                mySmallBodyTemp.PEmaloomega=row[1]
                mySmallBodyTemp.PEvelikoOmega=row[2]
                mySmallBodyTemp.PEi=row[3]
                mySmallBodyTemp.PEe=row[4]
                mySmallBodyTemp.PEq=row[5]
                mySmallBodyTemp.PEa=row[6]
                mySmallBodyTemp.PEM=row[7]
                mySmallBodyTemp.PEn=row[8]
                mySmallBodyTemp.PEQ=row[9]
                mySmallBodyTemp.PEP=row[10]
                mySmallBodyTemp.PEH=row[11]
                

                ####mySmallBodyTemp.SBPrintName()
                ####mySmallBodyTemp.SBPrint()

                MylistOfSB.append(SmallBody(row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],
                                      row[9],row[10],row[11]))
                # Check that there are exactly 11 numbers
                if len(numbers) == 11:
                    data_rows.append(DataRow(label, numbers))
                else:
                    print(f"Line {line_num} does not contain exactly 11 numbers: {row}")
            except ValueError:
                print(f"Line {line_num} contains non-numeric values in the numbers section: {row}")    
    return data_rows


file_path = 'Set005NEOs36524.csv'


data = load_data_from_csv(file_path)
#### print(data)
###### štampanje imena prvog tela
####print (MylistOfSB[1].SBPrintName())
###imamo napunjenu listu iz CVS fajla





#### filtriranje 
MylistOfSB = [obj for obj in MylistOfSB if float(obj.PEa) < 4]



### podela na test i trening 
# Split MylistOfSB into two halves
half = len(MylistOfSB) // 2  # Find the midpoint of the list
MylistOfSBTraining = MylistOfSB[:half]  # First half for training
MylistOfSBTest = MylistOfSB[half:]  # Second half for testing

# Print the sizes to verify
print(f"Training list size: {len(MylistOfSBTraining)}")
print(f"Test list size: {len(MylistOfSBTest)}")




#####SemiMajor Axis vs inclination

import matplotlib.pyplot as plt300

# Convert the data into arrays
xs = [obj.PEa for obj in MylistOfSB]
resX = [float(ele) for ele in xs]
ys = [obj.PEi for obj in MylistOfSB]
resY = [float(ele) for ele in ys]

# Add labels and title
plt300.xlabel("a")
plt300.ylabel("i")
plt300.title(" a vs i")
plt300.legend()
# Plot the data points
plt300.scatter(resX, resY, s=0.8, color='blue', label='Data points')
# Show the plot
plt300.grid(True)
plt300.show()


###### SemiMajor Axis vs eccentricity
import matplotlib.pyplot as plt300

# Convert the data into arrays
xs = [obj.PEa for obj in MylistOfSB]
resX = [float(ele) for ele in xs]
ys = [obj.PEe for obj in MylistOfSB]
resY = [float(ele) for ele in ys]

# Add labels and title
plt300.xlabel("a")
plt300.ylabel("e")
plt300.title(" a vs e")
plt300.legend()
# Plot the data points
plt300.scatter(resX, resY, s=3, color='blue', label='Data points')
# Show the plot
plt300.grid(True)
plt300.show()




#### 4 Kategorije
MylistOfAmor = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEq) > 1.017 and float(obj.PEq) <1.3]
MylistOfApollo = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEq) < 1.017 and float(obj.PEa) >1]
MylistOfAtens = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEa) < 1 and float(obj.PEQ) >0.983]
MylistOfApohele = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEa) < 1 and float(obj.PEQ) < 0.983]



MylistOfAmorTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) > 1.017 and float(obj.PEq) <1.3]
MylistOfApolloTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) < 1.017 and float(obj.PEa) >1]
MylistOfAtensTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) >0.983]
MylistOfApoheleTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) < 0.983]





countAmor   =len(MylistOfAmor)
countApollo =len(MylistOfApollo)
countAtens  =len(MylistOfAtens)
countApohele=len(MylistOfApohele)
countPocetno=len(MylistOfSB)
countUkupno=countAmor+countApollo+countAtens+countApohele

print ("Pocetni broj NEO tela: "+ str(countPocetno))
print ("Ukupno posle filtracije (a<4 AU) ): "+ str(countUkupno))



##### semimajor axis vs eccentricitya sve 4 kategorije

import matplotlib.pyplot as plt300

# Convert the data into arrays
xs1 = [obj.PEa for obj in MylistOfAmor]
resX1 = [float(ele) for ele in xs1]
ys1 = [obj.PEe for obj in MylistOfAmor]
resY1 = [float(ele) for ele in ys1]

xs2 = [obj.PEa for obj in MylistOfApollo]
resX2 = [float(ele) for ele in xs2]
ys2 = [obj.PEe for obj in MylistOfApollo]
resY2 = [float(ele) for ele in ys2]

xs3 = [obj.PEa for obj in MylistOfAtens]
resX3 = [float(ele) for ele in xs3]
ys3 = [obj.PEe for obj in MylistOfAtens]
resY3 = [float(ele) for ele in ys3]

xs4 = [obj.PEa for obj in MylistOfApohele]
resX4 = [float(ele) for ele in xs4]
ys4 = [obj.PEe for obj in MylistOfApohele]
resY4 = [float(ele) for ele in ys4]

# Add labels and title
plt300.xlabel("a")
plt300.ylabel("e")
plt300.title(" a vs e")
plt300.legend()
# Plot the data points
##############plt300.scatter(resX, resY, s=3, color='green', label='Data points')
plt300.scatter(resX1, resY1, s=2 ,c='red', label='Amor')
plt300.scatter(resX2, resY2, s=2 ,c='blue', label='Apollo')
plt300.scatter(resX3, resY3, s=2 ,c='green', label='Atens')
plt300.scatter(resX4, resY4, s=2 ,c='black', label='Apohele')
# Show the plot
plt300.grid(True)
plt300.show()




#### random forest


### random forest ML 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Prepare data
list1 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAmor]
list2 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApollo]
list3 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAtens]
list4 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApohele]

# Combine all data into one dataset with labels
X = np.array(list1 + list2 + list3 + list4)  # Features
y = np.array([0] * len(list1) + [1] * len(list2) + [2] * len(list3) + [3] * len(list4))  # Labels
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Predict on the test set
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Random Forest Classifier: {accuracy * 100:.2f}%")
# Plot decision boundaries and clusters
plt.figure(figsize=(12, 8))
# Create a mesh grid for decision boundary visualization
x_min, x_max = X[:, 0].min() , X[:, 0].max() 
y_min, y_max = X[:, 1].min() , X[:, 1].max() 
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
# Predict for each point in the mesh grid
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
#### Plot decision boundaries
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.contour(xx, yy, Z, colors='black', linewidths=3)  # Black borders for cluster boundaries
# Scatter plot for each cluster
labels = ['Amor', 'Apollo', 'Atens', 'Apohele']
colors = ['red', 'blue', 'green', 'purple']
for cluster in range(4):
    plt.scatter(
        X[y == cluster, 0],
        X[y == cluster, 1],
        label=f"{labels[cluster]}",
        alpha=0.6,
        color=colors[cluster]
    )
plt.title("Scatter Plot of Classified Clusters with Decision Boundaries")
plt.xlabel("a")
plt.ylabel("e")
plt.legend()
plt.grid(True)
plt.show()



#### moja provera preciznosti random foresta
MylistOfAmorTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) > 1.017 and float(obj.PEq) <1.3]
MylistOfApolloTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) < 1.017 and float(obj.PEa) >1]
MylistOfAtensTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) >0.983]
MylistOfApoheleTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) < 0.983]

import numpy as np
points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAmorTest])
predicted_clusters = clf.predict(points)
# Count the number of points in cluster 0
count_cluster_0 = np.sum(predicted_clusters == 0)
print(f"Number of points in cluster 0: {count_cluster_0}")
print("Length of MylistOfAmorTest is " + str(len(MylistOfAmorTest)))

points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApolloTest])
predicted_clusters = clf.predict(points)
# Count the number of points in cluster 0
count_cluster_1 = np.sum(predicted_clusters == 1)
print(f"Number of points in cluster 1: {count_cluster_1}")
print("Length of MylistOfApolloTest is " + str(len(MylistOfApolloTest)))


points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAtensTest])
predicted_clusters = clf.predict(points)
# Count the number of points in cluster 0
count_cluster_2 = np.sum(predicted_clusters == 2)
print(f"Number of points in cluster 2: {count_cluster_2}")
print("Length of MylistOfAtensTest is " + str(len(MylistOfAtensTest)))


points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApoheleTest])
predicted_clusters = clf.predict(points)
# Count the number of points in cluster 0
count_cluster_3 = np.sum(predicted_clusters == 3)
print(f"Number of points in cluster 3: {count_cluster_3}")
print("Length of MylistOfApoheleTest is " + str(len(MylistOfApoheleTest)))


sumForProcena = count_cluster_0 + count_cluster_1 + count_cluster_2 +count_cluster_3
print("Ukupni broj Tela koji je dobro određen " + str(sumForProcena) )
print("Ukupan broj tela za testiranje je " + str(len(MylistOfSBTest)))

print (" Tacnost procene je "+ str (float (sumForProcena /len(MylistOfSBTest)) ) )


















