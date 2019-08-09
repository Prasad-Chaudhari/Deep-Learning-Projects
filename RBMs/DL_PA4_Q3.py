import numpy as np
import pandas as pd
from tsne import tsne
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


directory = "./"

train_data = pd.read_csv(directory+'train.csv')
test_data = pd.read_csv(directory+'test.csv')

train_x = train_data.drop(['id', 'label'], axis = 1).values
train_y = train_data['label'].values
test_x = test_data.drop(['id', 'label'], axis=1).values
test_y = test_data['label'].values

train_x = np.where(train_x >= 127, 1, 0)
test_x = np.where(test_x >= 127, 1, 0)

h_n = 200
v_m = 784
W = np.random.randn(h_n, v_m)
b = np.random.randn(v_m)-2
c = np.random.randn(h_n)-2
k = 5
learning_rate = 0.001
norms = []
index = np.random.randint(0, train_x.shape[0])

plt.imshow(train_x[index].reshape(28,28))
plt.show()
plt.clf()

plt.figure(figsize=(20,20))

for i in range(1, 60001):

    x = train_x[i-1]

    v_t = np.copy(x)
    h_t = np.zeros(h_n)

    for _ in range(k):
        pro_samples = np.random.rand(h_n)
        probs = sigmoid(np.matmul(W, v_t) + c)
        h_t = np.where(probs < pro_samples, 1, 0)

        pro_samples = np.random.rand(v_m)
        probs = sigmoid(np.matmul(h_t.T, W) + b)
        v_t = np.where(probs < pro_samples, 1, 0)

    if i % 100 == 0:
        v_t_ = np.copy(train_x[index])
        for _ in range(100):
            pro_samples = np.random.rand(h_n)
            probs = sigmoid(np.matmul(W, v_t_) + c)
            h_t_ = np.where(probs < pro_samples, 1, 0)

            pro_samples = np.random.rand(v_m)
            probs = sigmoid(np.matmul(h_t_.T, W) + b)
            v_t_ = np.where(probs < pro_samples, 1, 0)
        ax = plt.subplot(8,8, i//100)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(v_t_.reshape(28,28))
        print(i/100)

    update_c_by_vd = sigmoid(np.matmul(W, x) + c)
    update_c_by_vt = sigmoid(np.matmul(W, v_t) + c)
    update_W_by_vd = np.outer(update_c_by_vd, x)
    update_W_by_vt = np.outer(update_c_by_vt, v_t)

    W = W + learning_rate * (update_W_by_vd - update_W_by_vt)
    b = b + learning_rate * (x - v_t)
    c = c + learning_rate * (update_c_by_vd - update_c_by_vt)

print("learning done!!!!")
plt.savefig("All_plots.png")
