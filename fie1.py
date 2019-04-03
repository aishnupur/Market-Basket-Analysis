import pandas as pd
import matplotlib as pl
import numpy as np

dataset1 = pd.read_csv('train.csv')

print('Total products in dataset:',dataset1['Upc'].nunique())
print('Total transactions in dataset:',dataset1['VisitNumber'].nunique())
print('Total Departments in walmart:',dataset1['DepartmentDescription'].nunique())


col = ['VisitNumber','Upc']
dataset2 = dataset1[col]

dataset3 = dataset2.groupby('VisitNumber')['Upc'].apply(list)

#deletes rows which have description as 'Discount'
for index, row in dataset1.iterrows():
    if row['Description'] == 'Discount':
        dataset1.drop(index,inplace =True)
    elif row.IsNull('Description'):
        print(row['InvoiceNo'])

for index, row in dataset1.iterrows():
    if row['Description'] == DBNull.Value:
        print(row['InvoiceNo'])
