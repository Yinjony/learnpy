class Employee: 
    empCount = 0 
    _protect=1      #单下划线，自己与子类
    __private=2     #双下划线，只能自己
    def __init__(self, name, salary):#self只是类的实例，每个类的方法里都要有self，但不是个形参
        self.name = name 
        self.salary = salary 
        Employee.empCount += 1 
    def displayCount(self):
        print("Total Employee %d" % Employee.empCount)
    def displayEmployee(self): 
        print("Name : ", self.name, ", Salary: ", self.salary)
    def __del__(self):              #析构函数，销毁垃圾文件
        class__name=self.__class__.__name__
        print("删除",class__name)

emp1 = Employee("Zara", 2000) 
emp2 = Employee("Manni", 5000) 
emp1.displayEmployee() 
emp2.displayEmployee() 
print("名称",Employee.__name__)

class Parent: # 定义父类
    parentAttr = 100
    def __init__(self):
        print("调用父类构造函数")
    def parentMethod(self):
        print("调用父类方法")
    def setAttr(self, attr):
        Parent.parentAttr = attr 
    def getAttr(self): 
        print("父类属性 :", Parent.parentAttr)
class Child(Parent): # 定义子类 可以继承多个类，A(B,C)
    def __init__(self): 
        print("调用子类构造方法") 
    def childMethod(self): 
        print("调用子类方法")
c = Child() # 实例化子类
c.childMethod() # 调用子类的方法 
c.parentMethod() # 调用父类方法 
c.setAttr(200) # 再次调用父类的方法 - 设置属性值 
c.getAttr() # 再次调用父类的方法 - 获取属性值
print(issubclass(Child,Parent))#判断是否是其子类

class Parent: # 定义父类
    def myMethod(self): 
        print('调用父类方法')
class Child(Parent): # 定义子类 
    def myMethod(self): 
        print('调用子类方法') 
c = Child() # 子类实例
c.myMethod() # 子类调用重写方法
#类的运算符重载？