import numpy
import numpy as np
import matplotlib

#结构化数据
dt = np.dtype([('age',np.int8)])
a = np.array([(10,),(20,),(30,)],dtype = dt)
print(a)
print(a['age'])

student = np.dtype([('name','S20'),('age','i1'),('marks','f4')])
b = np.array([('abc',21,50),('xyz',18,75)],dtype = student)

numpy.empty([3,2],dtype = int)
numpy.zeros((5,),dtype = int)
numpy.ones((5,),dtype = int)

x = [1,2,3]
a = np.asarray(x)
print(a)

s = b'HELLO WORLD'
a = np.frombuffer(s,dtype = 'S1')
print(a)

a = np.linspace(1,10,10,True,True)