import pandas as pd
import os


DATA = os.path.join(os.getcwd(), 'dataset')
GROUP = 2
data_path = "..//dataset//testing_data{}.txt".format(GROUP)
print(data_path)

freader = open(os.path.join(DATA, data_path), 'r')
records = freader.readlines()
records = list(map(lambda e: e.strip().split('\t'), records))
df = pd.DataFrame(records)
df.columns = ['f1', 'f2', 'f3', 'f4', 'f5', 'label']
df.info()
class_1 = df[df['label']=='1']
class_2 = df[df['label']=='2']

class_3 = df[df['label']=='3']
print('{}, {}, {}'.format(len(class_1), len(class_2), len(class_3)))