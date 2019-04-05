import pandas as pd
import matplotlib as pl
import numpy as np
from apyori import apriori
dataset1 = pd.read_csv('train.csv')

print('Total products in dataset:',dataset1['Upc'].nunique())
print('Total transactions in dataset:',dataset1['VisitNumber'].nunique())
print('Total Departments in walmart:',dataset1['DepartmentDescription'].nunique())
print('Total trip types:',dataset1['TripType'].nunique())


col = ['VisitNumber','Upc']
dataset2 = dataset1[col]

dataset3 = dataset2.groupby('VisitNumber')['Upc'].apply(list)

frequent_itemsets = apriori(dataset3, min_support=0.0006, min_confidence=0.5, min_lift=3, min_length=2)
association_results = list(frequent_itemsets)
print(association_results[0])  

for item in association_results:
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: ",items[0], " -> " ,items[1])

    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    

#deletes rows which have description as 'Discount'
for index, row in dataset1.iterrows():
    if row['Description'] == 'Discount':
        dataset1.drop(index,inplace =True)
    elif row.IsNull('Description'):
        print(row['InvoiceNo'])

for index, row in dataset1.iterrows():
    if row['Description'] == DBNull.Value:
        print(row['InvoiceNo'])

