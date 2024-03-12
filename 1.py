'''
将整个模块(somemodule)导入，格式为： import somemodule

从某个模块中导入某个函数,格式为： from somemodule import somefunction

从某个模块中导入多个函数,格式为： from somemodule import firstfunc, secondfunc, thirdfunc

将某个模块中的全部函数导入，格式为： from somemodule import *

将某个模块改名(改为s)，格式为：import somemodule as s
'''
s1="hello world"                    #循环
for i in s1:
    print(i)

for letter in 'Python':
    if letter == 'h':
        pass                        #pass语句单纯占位，是一个空语句，便于代码结构化
    if letter == 't':
        print('当前字母 :', letter)  #整洁吗
print("Good bye!")

a=3+4j                              #复数
ax=complex(1,2)
print(a,ax)

s2="HELLO"
print(s2*2)                         #重复打印，其他的似乎也可以

p1=(1,2,3)                          #元组tuple类型
p2=(3,4,5)
p4=("jojo",12)
p3=p1+p2
print(p3,max(p3),min(p3),p4)
x1=[1,2,3,3,4,5]

d1=dict()                           #字典dict类型,键不可变
d2={}
print(d1,len(d1),type(d1))
print(d2,len(d2),type(d2))
tinydict = {'Name': 'zhangsan', 'Age': 7, 'Class': 'First'}
del tinydict['Name']                # 删除键 'Name' 
tinydict.clear()
print(tinydict)                     # 清空字典 
del tinydict                        # 删除字典

fruit={'apple', 'orange', 'pear', 'orange', 'banana'}#集合set类型
print(fruit)
a=set('abcccd')
print("a=",a)
b=set('ad')
c1=a-b                              #各种逻辑运算操作         
c2=a|b
c3=a&b
c4=a^b 
print(c1,c2,c3,c4)
b1={1,2,3,4}
b2={3,4,5,6}
print(b1.intersection(b2),b1.union(b2),b1.difference(b2))
fruit=set(("apple", "orange", "pear"))
fruit.add("banana")                 #添加单个元素
print(fruit)
fruit.update({1,2,3})               #添加多个元素，列表，元组，字典等等，如果是update("banana")，添加的是一个集合("b","a","n")
print(fruit)
fruit.remove("apple")               #删除，若没有会出错
fruit.discard("orange")             #删除，若没有也不会出错
fruit.pop()                         #随机删除
fruit.clear()

squares = [x**2 for x in range(9) if x % 2 == 0]#列表推导式
squares = [x**2 if x % 2 == 0 else x + 3 for x in range(9)]#if else形式
#使用列表推导式创建一个列表 multiples_3，能够计算出 1 - 20 这 20 个整数中分别乘以 3 之后的结果。
multiple_3=[x/3 for x in range(1,21)]
print(multiple_3)
print(len(multiple_3))

#TODO 使用列表推导式创建新的列表 first_names，其中仅包含names中的名字（小写形式）。
# 需要会用到split()函数
names = ["Rick Sanchez", "Morty Smith", "Summer Smith", "Jerry Smith", "Beth Smith"]
first_name=[i.split(" ")[0].lower()  for i in names]
print(first_name)

#TODO 使用列表推导式创建一个 passed 的姓名列表，其中仅包含得分至少为 65 分的名字。
scores = {
             "Rick Sanchez": 70,
             "Morty Smith": 35,
             "Summer Smith": 82,
             "Jerry Smith": 23,
             "Beth Smith": 98
          }
passed=[i for i in scores if scores[i]>=65 ]
print(passed)