#filter()函数，这是一个高阶内置函数，接受函数和可迭代对象作为输入，并返回一个由可迭代对象中的特定元素（该函数针对该元素会返回 True）组成的迭代器。
cities = ["New York City", "Los Angeles", "Chicago", "Mountain View", "Denver", "Boston"]
a=lambda name:len(name)<10
short_cities = list(filter(a, cities))
print(short_cities)

#zip 返回一个将多个可迭代对象组合成一个元组序列的迭代器。每个元组都包含所有可迭代对象中该位置的元素。
letters = ['a', 'b', 'c']
nums = [1, 2, 3]
for letter, num in zip(letters, nums):
    print("{}: {}".format(letter, num))
#可以把zip组合后的迭代器创建为一个字典
cast=dict(zip(letters, nums))
print(cast)
#除了可以将两个列表组合到一起之外，还可以使用星号拆分列表。
some_list = [('a', 1), ('b', 2), ('c', 3)]
letters, nums = zip(*some_list)
print(letters,nums)
#转置
data = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11))
data_transpose = tuple(zip(*data))
print(data_transpose)

#enumerate函数
letters = ['a', 'b', 'c', 'd', 'e']
for i, letter in enumerate(letters):
    print(i, letter)