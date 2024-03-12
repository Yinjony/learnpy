x=[1,2,3,4]             #迭代器与生成器
i=iter(x)
j=iter(x)
print(next(i),next(i))
for t in j:
    print(t)            #可使用raise StopIteration，停止迭代

import sys 
def fibonacci(n):       # 生成器函数 - 斐波那契 
    a, b, counter = 0, 1, 0 
    while True: 
        if (counter > n): 
            return 
        yield a         #感觉相当于链表，生成一个一个迭代点，然后最终返回的是所有迭代点组成的迭代器
        a, b = b, a + b 
        counter += 1 
f = fibonacci(10)       # f是一个迭代器，由生成器返回生成
while True:
    try: 
        print (next(f), end=" ")
    except StopIteration: 
        sys.exit()


