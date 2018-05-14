import pandas as pd
import  numpy as np

sub1 = pd.read_csv('C:\Users\ZhangYue\Desktop\submissions\BI_LSTM_baseline.csv')
sub2 = pd.read_csv('C:\Users\ZhangYue\Desktop\submissions\glove200_baseline.csv')
sub3 = pd.read_csv('C:\Users\ZhangYue\Desktop\submissions\glove200_gru_baseline.csv')
sub4 = pd.read_csv('C:\Users\ZhangYue\Desktop\submissions\submission.csv')

sub1.columns
labels = sub1.columns[1:]

res = {}
res['id'] = sub1['id']
for label in labels:
    res[label] = np.mean([sub1[label], sub2[label], sub3[label], sub4[label]], axis=0)

len(res)
len(res['toxic'])

sub_stack = pd.DataFrame(res)
sub_stack.to_csv('sub_stack.csv', index=False)

res = {}
res['id'] = sub1['id']
for label in labels:
    res[label] = np.max([sub2[label], sub3[label], sub4[label]], axis=0)

sub_stack = pd.DataFrame(res)
sub_stack.to_csv('sub_stack_max.csv', index=False)

# sub2: .9793, .9789, .9840
weights = np.exp([.9793, .9789, .9840])
weights = weights/np.sum(weights)
res = {}
res['id'] = sub1['id']
for label in labels:
    res[label] = np.array(weights).dot(np.array([sub2[label], sub3[label], sub4[label]]))

sub_stack = pd.DataFrame(res)
sub_stack.to_csv('sub_stack_weights.csv', index=False)
