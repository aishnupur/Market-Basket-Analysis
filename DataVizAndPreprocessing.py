# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 18:22:25 2018

@author: sandh
"""

data = pd.read_csv("train.csv")
w_test_data = pd.read_csv("test 2 (1).csv")

len(data[data.TripType == 8].VisitNumber.unique())
data.count()


data.TripType.unique()
len(data.TripType.unique())
len(data.VisitNumber.unique())
data.VisitNumber.max()
data.VisitNumber.min()

data.Weekday.unique()

data.Upc.unique()

data.ScanCount.unique()

data.DepartmentDescription.unique()


len(data.DepartmentDescription.unique())

len(data.FinelineNumber.unique())

data.FinelineNumber.max()

data.FinelineNumber.min()


data[data.FinelineNumber == 0].count()

fineline_is_zero = data[data.FinelineNumber == 0]

fineline_is_zero[fineline_is_zero.ScanCount == 1].count()

fineline_is_zero[fineline_is_zero.ScanCount == -1].count()

data_fineline_department = data[["DepartmentDescription", "FinelineNumber"]]


# Dropping rows with missing values

data = data.dropna()
data.count()


data = data.replace("Monday", 1)
data = data.replace("Tuesday", 2)
data = data.replace("Wednesday", 3)
data = data.replace("Thursday", 4)
data = data.replace("Friday", 5)
data = data.replace("Saturday", 6)
data = data.replace("Sunday", 7)


data.head()


x = data.TripType.unique()
np.sort(x)

data_triptypes = data.drop_duplicates("VisitNumber")

x = data_triptypes["TripType"]
x = x.value_counts()

graph = x.plot(kind="bar", figsize=(10, 5), color="midnightblue")
graph.set_title("Number of Occurences by trip type")

type_3 = data[data.TripType == 3]
type_3_items = type_3[["TripType","DepartmentDescription"]]
type_3_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                        title="Type 3 ", color="midnightblue")
plt.xticks(fontsize=18)
plt.ylabel('ylabel', fontsize=16)

type_4 = data[data.TripType == 4]
type_4_items = type_4[["TripType","DepartmentDescription"]]
type_4_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, title="Type 4 Trips", color="midnightblue")

type_5 = data[data.TripType == 5]
type_5_items = type_5[["TripType","DepartmentDescription"]]
type_5_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 5 Trips", color="midnightblue")

type_6 = data[data.TripType == 6]
type_6_items = type_6[["TripType","DepartmentDescription"]]
type_6_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 6 Trips", color="midnightblue")


type_7 = data[data.TripType == 7]
type_7_items = type_7[["TripType","DepartmentDescription"]]
type_7_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 7 Trips", color="midnightblue")

type_8 = data[data.TripType == 8]
type_8_items = type_8[["TripType","DepartmentDescription"]]
type_8_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 8 Trips", color="midnightblue")

type_9 = data[data.TripType == 9]
type_9_items = type_9[["TripType","DepartmentDescription"]]
type_9_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 9 Trips", color="midnightblue")



type_12 = data[data.TripType == 12]
type_12_items = type_12[["TripType","DepartmentDescription"]]
type_12_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 12 Trips", color="midnightblue")

type_14 = data[data.TripType == 14]
type_14_items = type_14[["TripType","DepartmentDescription"]]
type_14_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 14 Trips", color="midnightblue")

type_15 = data[data.TripType == 15]
type_15_items = type_15[["TripType","DepartmentDescription"]]
x = type_15_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 15 Trips", color="midnightblue")


type_18 = data[data.TripType == 18]
type_18_items = type_18[["TripType","DepartmentDescription"]]
type_18_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 18 Trips", color="midnightblue")

type_19 = data[data.TripType == 19]
type_19_items = type_19[["TripType","DepartmentDescription"]]
x = type_19_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 19 Trips", color="midnightblue")

type_20 = data[data.TripType == 20]
type_20_items = type_20[["TripType","DepartmentDescription"]]
x = type_20_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Trip Type #20", color="midnightblue")
                                                              
                                
type_21 = data[data.TripType == 21]
type_21_items = type_21[["TripType","DepartmentDescription"]]
x = type_21_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 21 Trips", color="midnightblue")

type_22 = data[data.TripType == 22]
type_22_items = type_22[["TripType","DepartmentDescription"]]
x = type_22_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 22 Trips", color="midnightblue")

type_23 = data[data.TripType == 23]
type_23_items = type_23[["TripType","DepartmentDescription"]]
x = type_23_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 23 Trips", color="midnightblue")

type_24 = data[data.TripType == 24]
type_24_items = type_24[["TripType","DepartmentDescription"]]
x = type_24_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 24 Trips", color="midnightblue")

type_25 = data[data.TripType == 25]
type_25_items = type_25[["TripType","DepartmentDescription"]]
x = type_25_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 25 Trips", color="midnightblue")



type_26 = data[data.TripType == 26]
type_26_items = type_26[["TripType","DepartmentDescription"]]
x = type_26_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 26 Trips", color="midnightblue")


type_27 = data[data.TripType == 27]
type_27_items = type_27[["TripType","DepartmentDescription"]]
x = type_27_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 27 Trips", color="midnightblue")


type_28 = data[data.TripType == 28]
type_28_items = type_28[["TripType","DepartmentDescription"]]
x = type_28_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 28 Trips", color="midnightblue")

type_29 = data[data.TripType == 29]
type_29_items = type_29[["TripType","DepartmentDescription"]]
x = type_29_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 29 Trips", color="midnightblue")

type_30 = data[data.TripType == 30]
type_30_items = type_30[["TripType","DepartmentDescription"]]
x = type_30_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 30 Trips", color="midnightblue")

type_31 = data[data.TripType == 31]
type_31_items = type_31[["TripType","DepartmentDescription"]]
x = type_31_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 31 Trips", color="midnightblue")

type_32 = data[data.TripType == 32]
type_32_items = type_32[["TripType","DepartmentDescription"]]
x = type_32_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 32 Trips", color="midnightblue")

type_33 = data[data.TripType == 33]
type_33_items = type_33[["TripType","DepartmentDescription"]]
x = type_33_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 33 Trips", color="midnightblue")

type_34 = data[data.TripType == 34]
type_34_items = type_34[["TripType","DepartmentDescription"]]
x = type_34_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 34 Trips", color="midnightblue")

type_35 = data[data.TripType == 35]
type_35_items = type_35[["TripType","DepartmentDescription"]]
x = type_35_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 35 Trips", color="midnightblue")


type_36 = data[data.TripType == 36]
type_36_items = type_36[["TripType","DepartmentDescription"]]
x = type_36_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 36 Trips", color="midnightblue")

type_37 = data[data.TripType == 37]
type_37_items = type_37[["TripType","DepartmentDescription"]]
x = type_37_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 37 Trips", color="midnightblue")

type_38 = data[data.TripType == 38]
type_38_items = type_38[["TripType","DepartmentDescription"]]
x = type_38_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 38 Trips", color="midnightblue")


type_39 = data[data.TripType == 39]
type_39_items = type_39[["TripType","DepartmentDescription"]]
x = type_39_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 39 Trips", color="midnightblue")

type_40 = data[data.TripType == 40]
type_40_items = type_40[["TripType","DepartmentDescription"]]
x = type_40_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                                  title="Type 40 Trips", color="midnightblue")

type_41 = data[data.TripType == 41]
type_41_items = type_41[["TripType","DepartmentDescription"]]
x = type_41_items.DepartmentDescription.value_counts().head(10).plot(kind="bar", rot=45, 
                                                              title="Type 41 Trips", color="midnightblue")

type_42 = data[data.TripType == 42]
type_42_items = type_42[["TripType","DepartmentDescription"]]
x = type_42_items.DepartmentDescription.value_counts().head(10).plot(kind="bar", rot=45, 
                                                              title="Type 42 Trips", color="midnightblue")

type_43 = data[data.TripType == 43]
type_43_items = type_43[["TripType","DepartmentDescription"]]
x = type_43_items.DepartmentDescription.value_counts().head(10).plot(kind="bar", rot=45, 
                                                              title="Type 43 Trips", color="midnightblue")

type_44 = data[data.TripType == 44]
type_44_items = type_44[["TripType","DepartmentDescription"]]
x = type_44_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 44 Trips", color="midnightblue")

type_999 = data[data.TripType == 999]
type_999_items = type_999[["TripType","DepartmentDescription"]]
x = type_999_items.DepartmentDescription.value_counts().head().plot(kind="bar", rot=45, 
                                                              title="Type 999 Trips", color="midnightblue")

mytraingrouped = mytrain.groupby("TripType")
mytraingrouped = mytraingrouped.aggregate(np.mean)

mytraingrouped_categories = mytraingrouped.ix[:,4:72]
                       

mytestgrouped = mytest.groupby("TripType")
mytestgrouped = mytestgrouped.aggregate(np.mean)

mytestgrouped_categories = mytestgrouped.ix[:,4:72]

a4_dims = (12, 15)
fig, ax = plt.subplots(figsize=a4_dims)
seaborn.heatmap(ax=ax, data=mytrain.T.sort_index(), cmap = 'seismic', linecolor='lightgrey', linewidths=.00000000000000001)

ax.xaxis.tick_top()

plt.title('TripType',y=1.04)

a4_dims = (12, 15)
fig, ax = plt.subplots(figsize=a4_dims)
seaborn.heatmap(ax=ax, data=mytraingrouped_categories.T.sort_index(), cmap = 'seismic', linecolor='lightgrey', linewidths=.00000000000000001)

ax.xaxis.tick_top()

plt.title('TripType',y=1.04)


a4_dims = (12, 15)
fig, ax = plt.subplots(figsize=a4_dims)
seaborn.heatmap(ax=ax, data=mytestgrouped_categories.T.sort_index(), cmap = 'seismic', linecolor='lightgrey', linewidths=.00000000000000001)

ax.xaxis.tick_top()

plt.title('TripType',y=1.04)







