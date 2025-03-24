import numpy as np

l = np.load("april_eval/greedy_3_num_removed.npy", allow_pickle=True)
k=l[300:]
print(k)
print(30-np.mean(k))
print(np.std(k))