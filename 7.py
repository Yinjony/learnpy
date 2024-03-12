l=[8,5,2,4,3,6,5,5,1,4,5]
print("最多数字：",max(l,key=l.count),"次数：",l.count(max(l,key=l.count)))
