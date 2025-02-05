# -*- coding: utf-8 -*-
"""
16.11. 2024. - 1.2.2025.
author: Dragoljub Perišić
verzija 5 kategorija - final 1.0
"""

import copy

# definicja  SmallBody class
class SmallBody:
    def __init__(self, Desn, H, G, Epoch, M, Peri, Node, PEi, PEe, n, PEa, PEq, PEQ):
        self.Desn = Desn
        self.H = H
        self.G = G
        self.Epoch = Epoch
        self.M = M
        self.Peri = Peri
        self.Node = Node
        self.PEi = PEi
        self.PEe = PEe
        self.n = n
        self.PEa = PEa
        self.PEq = PEq
        self.PEQ = PEQ

# funnkcija koja učitava objekte slammbody u listu MyListOfSB

def loadSmallBodies(input_file):
    MyListOfSB = []
    try:
        with open(input_file, 'r') as infile:
            header = infile.readline()  # preskoči heder

            for line in infile:
                columns = line.split()
                if len(columns) >= 13:
                    try:
                        # kreira  SmallBody objekat
                        sb = SmallBody(
                            Desn=columns[0],
                            H=float(columns[1]),
                            G=float(columns[2]),
                            Epoch=columns[3],
                            M=float(columns[4]),
                            Peri=float(columns[5]),
                            Node=float(columns[6]),
                            PEi=float(columns[7]),
                            PEe=float(columns[8]),
                            n=float(columns[9]),
                            PEa=float(columns[10]),
                            PEq=float(columns[11]),
                            PEQ=float(columns[12])
                        )
                        MyListOfSB.append(sb)
                    except ValueError:
                        continue  # Bitno : preskače  linije u slučaju loše konverzije u float
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    return MyListOfSB




# Glavni deo
input_file = "input40K.txt"

### polazni fajl je sa sajta MPCORB.DAT sa sajta https://minorplanetcenter.net/data 21.dec.2024.
### fajl input40K.txt sadrži 40123 objekata koji je generisan pomoću pomoćnih programa koji su navedeni
### ispod glavnog programa .. fajl je generisan tako da sadrži tela u svih 5 kategorija





MyListOfSB = loadSmallBodies(input_file)

# Create the five lists
MylistOfAmor = [copy.deepcopy(obj) for obj in MyListOfSB if 1.017 < float(obj.PEq) < 1.3]
MylistOfApollo = [copy.deepcopy(obj) for obj in MyListOfSB if float(obj.PEq) < 1.017 and float(obj.PEa) > 1]
MylistOfAtens = [copy.deepcopy(obj) for obj in MyListOfSB if float(obj.PEa) < 1 and float(obj.PEQ) > 0.983]
MylistOfApohele = [copy.deepcopy(obj) for obj in MyListOfSB if float(obj.PEa) < 1 and float(obj.PEQ) < 0.983]
MylistofNonNEO = [copy.deepcopy(obj) for obj in MyListOfSB if float(obj.PEq) > 1.3]

# Count the objects in each list
countAmor = len(MylistOfAmor)
countApollo = len(MylistOfApollo)
countAtens = len(MylistOfAtens)
countApohele = len(MylistOfApohele)
countNonNEO = len(MylistofNonNEO)


countPocetno=len(MyListOfSB)
countUkupno=countAmor+countApollo+countAtens+countApohele+countNonNEO


# Print the results
print(f"countAmor: {countAmor}")
print(f"countApollo: {countApollo}")
print(f"countAtens: {countAtens}")
print(f"countApohele: {countApohele}")
print(f"countNonNEO: {countNonNEO}")
print(f"Sum: {countAmor + countApollo + countAtens + countApohele + countNonNEO}")

### ---- dole je ML deo




#### filtriranje 
MyListOfSB = [obj for obj in MyListOfSB if float(obj.PEa) < 6]



### podela na test i trening 
# MylistOfSB lista je podeljenja na dva jednaka dela
half = len(MyListOfSB) // 2 
print(f"Ukupno tela: {len(MyListOfSB)}")
MylistOfSBTraining = MyListOfSB[:half]  # prva polovina za trening 
MylistOfSBTest = MyListOfSB[half:]  # druga polovina za test

print(f"Training list size: {len(MylistOfSBTraining)}")
print(f"Test list size: {len(MylistOfSBTest)}")




#####podela na test i trening svake liste ponaosob


half = len(MylistOfAmor) // 2  
print(f"Ukupno tela: {len(MylistOfAmor)}")
MylistOfAmorTraining = MylistOfAmor[:half]  
MylistOfAmorTest = MylistOfAmor[half:]  
print(f"Training list size: {len(MylistOfAmorTraining)}")
print(f"Test list size: {len(MylistOfAmorTest)}")

half = len(MylistOfApollo) // 2  
print(f"Ukupno tela: {len(MylistOfApollo)}")
MylistOfApolloTraining = MylistOfApollo[:half]  
MylistOfApolloTest = MylistOfApollo[half:]  
print(f"Training list size: {len(MylistOfApolloTraining)}")
print(f"Test list size: {len(MylistOfApolloTest)}")


half = len(MylistOfAtens) // 2  
print(f"Ukupno tela: {len(MylistOfAtens)}")
MylistOfAtensTraining = MylistOfAtens[:half]  
MylistOfAtensTest = MylistOfAtens[half:]  
print(f"Training list size: {len(MylistOfAtensTraining)}")
print(f"Test list size: {len(MylistOfAtensTest)}")


half = len(MylistOfApohele) // 2  
print(f"Ukupno tela: {len(MylistOfApohele)}")
MylistOfApoheleTraining = MylistOfApohele[:half]  
MylistOfApoheleTest = MylistOfApohele[half:]  
print(f"Training list size: {len(MylistOfApoheleTraining)}")
print(f"Test list size: {len(MylistOfApoheleTest)}")

half = len(MylistofNonNEO) // 2  
print(f"Ukupno tela: {len(MylistofNonNEO)}")
MylistofNonNEOTraining = MylistofNonNEO[:half]  
MylistofNonNEOTest = MylistofNonNEO[half:]  
print(f"Training list size: {len(MylistofNonNEOTraining)}")
print(f"Test list size: {len(MylistofNonNEOTest)}")




granica =10 ### 10 astronomskih jedinica za veliku poluosu

MylistOfAmorTraining = [obj for obj in MylistOfAmorTraining if float(obj.PEa) < granica]
MylistOfAmorTest = [obj for obj in MylistOfAmorTest if float(obj.PEa) < granica]

MylistOfApolloTraining = [obj for obj in MylistOfApolloTraining if float(obj.PEa) < granica]
MylistOfApolloTest = [obj for obj in MylistOfApolloTest if float(obj.PEa) < granica]

MylistOfAtensTraining = [obj for obj in MylistOfAtensTraining if float(obj.PEa) < granica]
MylistOfAtensTest = [obj for obj in MylistOfAtensTest if float(obj.PEa) < granica]

MylistOfApoheleTraining = [obj for obj in MylistOfApoheleTraining if float(obj.PEa) < granica]
MylistOfApoheleTest = [obj for obj in MylistOfApoheleTest if float(obj.PEa) < granica]

MylistofNonNEOTraining = [obj for obj in MylistofNonNEOTraining if float(obj.PEa) < granica]
MylistofNonNEOTest = [obj for obj in MylistofNonNEOTest if float(obj.PEa) < granica]





#####SemiMajor Axis vs inclination  a vs i

import matplotlib.pyplot as plt

# Convert the data into arrays
xs = [obj.PEa for obj in MyListOfSB if obj.PEa is not None]
resX = [float(ele) for ele in xs]
ys = [obj.PEi for obj in MyListOfSB if obj.PEi is not None]
resY = [float(ele) for ele in ys]

# minimum i maximum za x and y 
min_x, max_x = min(resX), max(resX)
min_y, max_y = min(resY), max(resY)

print(f"X range: {min_x} to {max_x}")
print(f"Y range: {min_y} to {max_y}")

# Adjust the scatter plot size dynamically based on the range
marker_size = max(1, 100 / len(resX))  # Dynamic marker size; smaller if too many points

# Labele i naslov
plt.xlabel("a")
plt.ylabel("i")
plt.title("a vs i")
plt.grid(True)

# crtanje tačaka
plt.scatter(resX, resY, s=marker_size, color='blue', label='Data points')

# legenda
plt.legend()

# granice za ose  na osnovu Set data ranges
plt.xlim(min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x))  
plt.ylim(min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y))  

# crta plot
plt.show()


###### SemiMajor Axis vs eccentricity (a vs e)
import matplotlib.pyplot as plt300


xs = [obj.PEa for obj in MyListOfSB]
resX = [float(ele) for ele in xs]
ys = [obj.PEe for obj in MyListOfSB]
resY = [float(ele) for ele in ys]

plt300.xlabel("a")
plt300.ylabel("e")
plt300.title(" a vs e")
plt300.legend()
# crtanje tačaka
plt300.scatter(resX, resY, s=3, color='blue', label='Data points')
# prikaži plot
plt300.grid(True)
plt300.show()




#### 5 Kategorija
'''
MylistOfAmor = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEq) > 1.017 and float(obj.PEq) <1.3]
MylistOfApollo = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEq) < 1.017 and float(obj.PEa) >1]
MylistOfAtens = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEa) < 1 and float(obj.PEQ) >0.983]
MylistOfApohele = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEa) < 1 and float(obj.PEQ) < 0.983]
MylistofNonNEO  = [copy.deepcopy(obj) for obj in MylistOfSBTraining if float(obj.PEq) > 1.3 ]

MylistOfAmorTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) > 1.017 and float(obj.PEq) <1.3]
MylistOfApolloTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) < 1.017 and float(obj.PEa) >1]
MylistOfAtensTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) >0.983]
MylistOfApoheleTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) < 0.983]
MylistofNonNEOTest  = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) > 1.3 ]
'''

countAmor   =len(MylistOfAmor)
countApollo =len(MylistOfApollo)
countAtens  =len(MylistOfAtens)
countApohele=len(MylistOfApohele)
countNonNEO =len(MylistofNonNEO)

countAmorTest   =len(MylistOfAmorTest)
countApolloTest =len(MylistOfApolloTest)
countAtensTest  =len(MylistOfAtensTest)
countApoheleTest=len(MylistOfApoheleTest)
countNonNEOTest =len(MylistofNonNEOTest)

countPocetno=len(MyListOfSB)
countUkupno=countAmor+countApollo+countAtens+countApohele+countNonNEO


print ("Pocetni broj NEO tela: "+ str(countPocetno))
print ("Ukupno posle filtracije (a<4 AU) ): "+ str(countUkupno))

print ("countAmor: count  "+ str(countAmor))
print ("countApollo: count  "+ str(countApollo))
print ("countAtens: count  "+ str(countAtens))
print ("countApohele: count  "+ str(countApohele))
print ("MyListofNonNEO: count  "+ str(countNonNEO))

print ("countAmorTest: count  "+ str(countAmorTest))
print ("countApolloTest: count  "+ str(countApolloTest))
print ("countAtensTest: count  "+ str(countAtensTest))
print ("countApoheleTest: count  "+ str(countApoheleTest))
print ("MyListofNonNEOTest: count  "+ str(countNonNEOTest))








##### semimajor axis vs eccentricitya svih 5 kategorija

import matplotlib.pyplot as plt300

# konverzija lista u niz
xs1 = [obj.PEa for obj in MylistOfAmorTraining]
resX1 = [float(ele) for ele in xs1]
ys1 = [obj.PEe for obj in MylistOfAmorTraining]
resY1 = [float(ele) for ele in ys1]

xs2 = [obj.PEa for obj in MylistOfApolloTraining]
resX2 = [float(ele) for ele in xs2]
ys2 = [obj.PEe for obj in MylistOfApolloTraining]
resY2 = [float(ele) for ele in ys2]

xs3 = [obj.PEa for obj in MylistOfAtensTraining]
resX3 = [float(ele) for ele in xs3]
ys3 = [obj.PEe for obj in MylistOfAtensTraining]
resY3 = [float(ele) for ele in ys3]

xs4 = [obj.PEa for obj in MylistOfApoheleTraining]
resX4 = [float(ele) for ele in xs4]
ys4 = [obj.PEe for obj in MylistOfApoheleTraining]
resY4 = [float(ele) for ele in ys4]

xs5 = [obj.PEa for obj in MylistofNonNEOTraining]
resX5 = [float(ele) for ele in xs5]
ys5 = [obj.PEe for obj in MylistofNonNEOTraining]
resY5 = [float(ele) for ele in ys5]


plt300.xlabel("a")
plt300.ylabel("e")
plt300.title(" a vs e")
plt300.legend()
# crtanje tačaka
plt300.scatter(resX1, resY1, s=2 ,c='red', label='Amor')
plt300.scatter(resX2, resY2, s=2 ,c='blue', label='Apollo')
plt300.scatter(resX3, resY3, s=2 ,c='green', label='Atens')
plt300.scatter(resX4, resY4, s=2 ,c='black', label='Apohele')
plt300.scatter(resX5, resY5, s=2 ,c='yellow', label='NonNEO')
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
list1 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAmorTraining]
list2 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApolloTraining]
list3 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAtensTraining]
list4 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApoheleTraining]
list5 = [(float(obj.PEa), float(obj.PEe)) for obj in MylistofNonNEOTraining]

# Combine all data into one dataset with labels
X = np.array(list1 + list2 + list3 + list4 + list5)  # Features
y = np.array([0] * len(list1) + [1] * len(list2) + [2] * len(list3) + [3] * len(list4)+ [4] * len(list5))  # Labels
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
# Predict on the test set
y_pred = clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Tačnost Random Forest Klasifikatora: {accuracy * 100:.2f}%")
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
###############plt.contour(xx, yy, Z, colors='black', linewidths=3)  # Black borders for cluster boundaries
# Scatter plot for each cluster
labels = ['Amor', 'Apollo', 'Atens', 'Apohele', 'NonNEO']
colors = ['red', 'blue', 'green', 'purple', 'yellow']
for cluster in range(5):
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
'''
MylistOfAmorTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) > 1.017 and float(obj.PEq) <1.3]
MylistOfApolloTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) < 1.017 and float(obj.PEa) >1]
MylistOfAtensTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) >0.983]
MylistOfApoheleTest = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEa) < 1 and float(obj.PEQ) < 0.983]
MylistofNonNEOTest  = [copy.deepcopy(obj) for obj in MylistOfSBTest if float(obj.PEq) > 1.3 ]
'''

#### provera

import numpy as np

points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAmorTest])
predicted_clusters = clf.predict(points)
count_cluster_0 = np.sum(predicted_clusters == 0)
print(f"Broj tačaka u klasteru 0: {count_cluster_0}")
print("Length of MylistOfAmorTest is " + str(len(MylistOfAmorTest)))

points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApolloTest])
predicted_clusters = clf.predict(points)
count_cluster_1 = np.sum(predicted_clusters == 1)
print(f"Broj tačaka u klasteru 1: {count_cluster_1}")
print("Length of MylistOfApolloTest is " + str(len(MylistOfApolloTest)))

points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfAtensTest])
predicted_clusters = clf.predict(points)
count_cluster_2 = np.sum(predicted_clusters == 2)
print(f"Broj tačaka u klasteru 2: {count_cluster_2}")
print("Length of MylistOfAtensTest is " + str(len(MylistOfAtensTest)))

points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistOfApoheleTest])
predicted_clusters = clf.predict(points)
count_cluster_3 = np.sum(predicted_clusters == 3)
print(f"Broj tačaka u klasteru 3: {count_cluster_3}")
print("Length of MylistOfApoheleTest is " + str(len(MylistOfApoheleTest)))

points = np.array([(float(obj.PEa), float(obj.PEe)) for obj in MylistofNonNEOTest])
predicted_clusters = clf.predict(points)
count_cluster_4 = np.sum(predicted_clusters == 4)
print(f"Broj tačaka u klasteru 4: {count_cluster_4}")
print("Length of MylistofNonNEOTest is " + str(len(MylistofNonNEOTest)))

sumForProcena = count_cluster_0 + count_cluster_1 + count_cluster_2 +count_cluster_3 + count_cluster_4
print("Ukupni broj Tela koji je dobro određen " + str(sumForProcena) )
print("Ukupan broj tela za testiranje je " + str(len(MylistOfSBTest)))

print (" Tačnost procene je "+ str (float (sumForProcena /len(MylistOfSBTest)) ) )



######################################################
#### pomoćni programi za pripremu ulaznih podataka
######################################################

## 01 isecanje kolona
'''

import os

# Program to process a file and retain only the first 11 columns, align them, and save to a new file

input_file = "uzorak za chatgptSVI10ktxtSVE.txt"  # Input file name
output_file = "out.txt"  # Output file name

try:
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Split the line into columns and extract the first 11
            columns = line.split()
            selected_columns = columns[:11]

            # Align columns by joining with spaces
            aligned_line = ' '.join(f"{col:<12}" for col in selected_columns).rstrip()
            outfile.write(aligned_line + '\n')

    print(f"Processing complete. Output written to {output_file}")
except FileNotFoundError:
    print(f"Error: File {input_file} not found.")
except Exception as e:
    print(f"An error occurred: {e}")
    
    
uzorak fajla posle isecanje

Des'n        H            G            Epoch        M            Peri.        Node         Incl.        e            n            a
00001        3.34         0.15         K24AH        145.84905    73.28579     80.25414     10.58790     0.0791840    0.21418047   2.7666197
00002        4.11         0.15         K24AH        126.06756    310.89226    172.90614    34.92186     0.2304384    0.21374870   2.7703442
00003        5.18         0.15         K24AH        127.32529    247.81975    169.83829    12.98815     0.2561092    0.22588717   2.6701869
00004        3.25         0.15         K24AH        278.02316    151.67629    103.70474    7.14398      0.0900011    0.27169443   2.3609252
00005        6.99         0.15         K243V        350.98291    359.23648    141.46063    5.35914      0.1872507    0.23840266   2.5758979
00006        5.61         0.15         K24AH        248.27292    239.64813    138.61983    14.73436     0.2024234    0.26082989   2.4260392    
    

'''




## 02 dodavanje q i Q kolona
'''
import os

# Program to process a file, calculate q and Q, and save the results in a formatted output file

def calculate_q_and_Q(input_file, output_file):
    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            header = infile.readline()
            outfile.write(header.strip() + "    q           Q\n")  # Write header with new columns

            for line in infile:
                # Split the line into columns
                columns = line.split()

                # Ensure the line has at least 11 columns to proceed
                if len(columns) >= 11:
                    try:
                        # Extract eccentricity (e) and semi-major axis (a)
                        e = float(columns[8])
                        a = float(columns[10])

                        # Calculate q and Q
                        q = a * (1 - e)
                        Q = a * (1 + e)

                        # Add q and Q to the line with alignment
                        formatted_line = ' '.join(f"{col:<12}" for col in columns)
                        formatted_line += f"{q:<12.6f} {Q:<12.6f}\n"

                        # Write the formatted line to the output file
                        outfile.write(formatted_line)
                    except ValueError:
                        # Skip lines where a cannot be converted to float
                        continue

        print(f"Processing complete. Output written to {output_file}")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Input and output file names
input_file = "GlaniIN.txt"
output_file = "outqanaQ.txt"

# Execute the processing function
calculate_q_and_Q(input_file, output_file)


fajl sa dodatim q i Q kolonama - u uzorak

Des'n        H            G            Epoch        M            Peri.        Node         Incl.        e            n            a           q            Q
00001        3.34         0.15         K24AH        145.84905    73.28579     80.25414     10.58790     0.0791840    0.21418047   2.7666197   2.547548     2.985692    
00002        4.11         0.15         K24AH        126.06756    310.89226    172.90614    34.92186     0.2304384    0.21374870   2.7703442   2.131951     3.408738    
00003        5.18         0.15         K24AH        127.32529    247.81975    169.83829    12.98815     0.2561092    0.22588717   2.6701869   1.986327     3.354046    
00004        3.25         0.15         K24AH        278.02316    151.67629    103.70474    7.14398      0.0900011    0.27169443   2.3609252   2.148439     2.573411    
00005        6.99         0.15         K243V        350.98291    359.23648    141.46063    5.35914      0.1872507    0.23840266   2.5758979   2.093559     3.058237    
00006        5.61         0.15         K24AH        248.27292    239.64813    138.61983    14.73436     0.2024234    0.26082989   2.4260392   1.934952     2.917126    


'''


## 03 brojanje 5 kategorija


'''
import os

# Program to categorize lines based on q, Q, a, and e values

def categorize_objects(input_file):
    # Initialize counters
    Amor = 0
    Apollo = 0
    Atens = 0
    Apohele = 0
    NonNEO = 0

    try:
        with open(input_file, 'r') as infile:
            header = infile.readline()  # Skip the header

            total_lines = 0

            for line in infile:
                total_lines += 1
                columns = line.split()

                # Ensure the line has at least 13 columns to proceed
                if len(columns) >= 13:
                    try:
                        # Extract necessary values as floats
                        e = float(columns[8])
                        a = float(columns[10])
                        q = float(columns[11])
                        Q = float(columns[12])

                        # Categorize based on conditions
                        if 1.017 < q < 1.3:
                            Amor += 1
                        elif q < 1.017 and a > 1:
                            Apollo += 1
                        elif a < 1 and Q > 0.983:
                            Atens += 1
                        elif a < 1 and Q < 0.983:
                            Apohele += 1
                        elif q > 1.3:
                            NonNEO += 1
                    except ValueError:
                        # Skip lines where conversion fails
                        continue

            # Print results
            print(f"Number of lines: {total_lines}")
            print(f"Number of Amor: {Amor}")
            print(f"Number of Apollo: {Apollo}")
            print(f"Number of Atens: {Atens}")
            print(f"Number of Apohele: {Apohele}")
            print(f"Number of NonNEO: {NonNEO}")
            print(f"Sum of all categories: {Amor + Apollo + Atens + Apohele + NonNEO}")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Input file name
input_file = "aeqQinputSVI.txt"

# Execute the categorization function
categorize_objects(input_file)

'''






### 04 pravljenje uzorka od 40000+ objekata
'''
import os

# Program to categorize lines based on q, Q, a, and e values, and write results to OUT.txt

def categorize_and_write(input_file, output_file):
    # Initialize counters
    Amor = 0
    Apollo = 0
    Atens = 0
    Apohele = 0
    NonNEO = 0
    WrittenLines = 0

    try:
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            header = infile.readline()  # Read header line
            outfile.write(header)  # Write header to output file

            total_lines = 0

            for line in infile:
                total_lines += 1
                columns = line.split()

                # Ensure the line has at least 13 columns to proceed
                if len(columns) >= 13:
                    try:
                        # Extract necessary values as floats
                        e = float(columns[8])
                        a = float(columns[10])
                        q = float(columns[11])
                        Q = float(columns[12])

                        # Categorize based on conditions and write lines if applicable
                        if 1.017 < q < 1.3:
                            Amor += 1
                            outfile.write(line)
                            WrittenLines += 1
                        elif q < 1.017 and a > 1:
                            Apollo += 1
                            outfile.write(line)
                            WrittenLines += 1
                        elif a < 1 and Q > 0.983:
                            Atens += 1
                            outfile.write(line)
                            WrittenLines += 1
                        elif a < 1 and Q < 0.983:
                            Apohele += 1
                            outfile.write(line)
                            WrittenLines += 1
                        elif q > 1.3 and WrittenLines < 40000:
                            NonNEO += 1
                            outfile.write(line)
                            WrittenLines += 1
                    except ValueError:
                        # Skip lines where conversion fails
                        continue

            # Print results
            print(f"Number of lines: {total_lines}")
            print(f"Number of Amor: {Amor}")
            print(f"Number of Apollo: {Apollo}")
            print(f"Number of Atens: {Atens}")
            print(f"Number of Apohele: {Apohele}")
            print(f"Number of NonNEO: {NonNEO}")
            print(f"Number of WrittenLines: {WrittenLines}")
            print(f"Sum of all categories: {Amor + Apollo + Atens + Apohele + NonNEO}")

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Input and output file names
input_file = "aeqQinputSVI.txt"
output_file = "OUT.txt"

# Execute the categorization function
categorize_and_write(input_file, output_file)
'''







