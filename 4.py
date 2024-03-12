fo = open("foo.txt", "r")
print("文件名: ", fo.name)
print("是否已关闭 : ", fo.closed)
print("访问模式 : ", fo.mode)
#fo.write("hello world")
s=fo.read(11)
print("当前文件的位置：",fo.tell())     #位置
print(s)
fo.seek(0,0)                           #前者是移动字节数，后者是从哪里开始
s=fo.read(11)
print(s)
camelot_lines = []
for line in f:
    camelot_lines.append(line.strip())#for循环行访问
print(camelot_lines)
fo.close()


with open('my_path/my_file.txt', 'r') as f:#缩进部分结束自动关闭
    file_data = f.read()
'''此外还有os模块方法'''
