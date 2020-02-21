import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# find the parameters (a,b)=(1,1) from Rosenbrock function
# f = 100*( pow((x2-x1*x1),2)+pow((x3-x2*x2),2) ) + (a-x1)*(a-x1) + (b-x2)*(b-x2)
# three sets of data: x1, x2, x3
def rosenbrock(x1, x2, x3):
    f = 100*( pow((x2-x1*x1),2)+pow((x3-x2*x2),2) ) + (1-x1)*(1-x1) + (1-x2)*(1-x2)
    return f

value = []
count = 0
wid = 3# x:(-3,3) 
# model data
for x1 in range(-wid,wid):
    for x2 in range(-wid,wid):
        for x3 in range(-wid,wid):
            z = rosenbrock(x1,x2,x3)
            #print "f(",x1,",",x2,",",x3,")=",z
            print x1,",",x2,",",x3,",",z
            value.append(z)
            count = count + 1

#v = np.array(value)
#plt.style.use('classic')
#plt.hist(value,bins = count)
#plt.show()

# Levenberg-Marquadt Method
alambda = 0.01
alpha = np.array(np.zeros(3))
beta = np.array(np.zeros(3))








