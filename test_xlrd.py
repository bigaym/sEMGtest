# 测试肌电信号及其导数之间关系
import numpy as np
import xlrd
import matplotlib.pyplot as plt

filename = "data\\上坡\\excel\\1.xlsx"
#加载文件
data = xlrd.open_workbook(filename)
#获取所有工作表名称
sheet_names = data.sheet_names()
print(sheet_names[0])
#获取工作表
table = data.sheet_by_name(sheet_names[0])
#获取名称、行列数
name = table.name
rowNum = table.nrows
colNum = table.ncols
#获取单元格内容的3种方式
i = 2
j = 0
value1 = table.cell(i,j).value
value2 = table.cell_value(i,j)
value3 = table.row(i)[j].value
print(value1,"\t",value2, "\t",value3)
print(rowNum,colNum)
print(table.row_values(0))
#获取单元格类型
print(table.cell(i,j).ctype)
print(type(table.cell(i,j)))