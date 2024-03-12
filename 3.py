import time
t=time.time()
print(t)                                                          # 格式化成YYYY-MM-DD HH:MM:SS形式
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))      # 格式化成
print (time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))   
a = "Sat Mar 28 22:24:24 2016" 
print (time.mktime(time.strptime(a,"%a %b %d %H:%M:%S %Y")))      # 将格式字符串转换为时间戳

def func(x):
   x = 10                                                         # 不可变对象，元组，字符串之类的，不会改变实参 
b = 2
func(b)
print(b)  
