# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 01:07:27 2024

@author: drago
"""
##   001 print
print ("Hello world")


## 002 dodela
num =34



## 003 if else
if (num>12):
    print("Number is good")
else: 
    print ("number is bad")
    
    
    
##  004 definicja funkcije    
    
def hello():
    return "hello world"
    
print (hello())
    
    
    
## 005 niz     
def demo(x):
    for i in range(5):
        print("i = {}, x = {}".format(i, x))
    x = x + 1

demo(0)
       

    


## 006 write to file ... read from file

import numpy as geek 
  
a = geek.array(([i + j for i in range(3)  
                       for j in range(3)])) 
# a is printed. 
print("a is:") 
print(a) 
  
geek.save('geekfile', a) 
print("the array is saved in the file geekfile.npy") 
  
# the array is saved in the file geekfile.npy  
b = geek.load('geekfile.npy') 
  
# the array is loaded into b 
print("b is:") 
print(b) 
  
# b is printed from geekfile.npy 
print("b is printed from geekfile.npy") 

##   end of write to file read from file



## 007 read write txt file 


f = open("demofile2.txt", "a")
f.write("Now the file has more content!")
f.close()

#open and read the file after the appending:
f = open("demofile2.txt", "r")
print(f.read())

### end of read write txt file 


## 008 mat plot lib plt.plot

       
import matplotlib.pyplot as plt
plt.plot(range(10), 'o')


## enf of mat plot lib 




"""  multi line comment

append()	Adds an element at the end of the list
clear()	Removes all the elements from the list
copy()	Returns a copy of the list
count()	Returns the number of elements with the specified value
extend()	Add the elements of a list (or any iterable), to the end of the current list
index()	Returns the index of the first element with the specified value
insert()	Adds an element at the specified position
pop()	Removes the element at the specified position
remove()	Removes the first item with the specified value
reverse()	Reverses the order of the list
sort()	Sorts the list

"""


## 009 array len , elements

myniz= [1,2,3,6,7,3,2,8,9,0]

f = open("mydata.txt", "w")

f.write(str(len(myniz))+"\n")  ## write count
for i in range(len(myniz)):
        x = myniz[i]
        f.write(str(x)+"\n")
        print (x)

f.close()


### 010 load array from file
myx = [0] * 10
mystr=""
myfile = open("mydata.txt", "r")
mystr= myfile.readline()
count=int(mystr)
print ("Mycount "+ str(count))
print("Using for loop")
for i in range(count):
       myx[i]=int(myfile.readline())   
       print("brojevi  " + str(myx[i]))
myfile.close()    

## --------------------------------------------------------




""" direktno iz fajla
import matplotlib.pyplot as plt 
import numpy as np
dataArray1= np.load(r'/home/user/Desktop/OutFileTraces.npy')
print(dataArray1)
plt.plot(dataArray1.T )
plt.show()
"""

### 011 moj array u plot 
import matplotlib.pyplot as plt 
plt.plot(myx )
plt.show()


import matplotlib.pyplot as plt2 
plt2.plot(myx )
plt2.show()



## ----------------------------------------------


### 012 generisanje sinusoide


import math

mynizSin= [0] *360

for i in range(len(mynizSin)):
               
    mynizSin[i] = 20 *math.sin((i/len(mynizSin)*2.00*math.pi))


f = open("mydataSin.txt", "w")

f.write(str(len(mynizSin))+"\n")  ## write count
for i in range(len(mynizSin)):
        x = mynizSin[i]
        f.write(str(x)+"\n")
##########        print (x)

f.close()


import matplotlib.pyplot as plt3
plt3.plot(mynizSin )
plt3.show()


### 013 generisanje kosinusoide

import math

mynizCos= [0] *360

for i in range(len(mynizCos)):
               
    mynizCos[i] = 20 *math.cos((i/len(mynizCos)*2.00*math.pi))


f = open("mydataCos.txt", "w")

f.write(str(len(mynizCos))+"\n")  ## write count
for i in range(len(mynizCos)):
        x = mynizCos[i]
        f.write(str(x)+"\n")
#########        print (x)

f.close()


import matplotlib.pyplot as plt4
plt4.plot(mynizCos )
plt4.show()




import matplotlib.pyplot as plt10
fig =plt10.figure(figsize=(6,6))
plt10.scatter(mynizSin, mynizCos)
plt10.title("krug")
plt10.show()
fig.savefig("Krug.png")



