import numpy as np
import pandas as pd
from tsne import tsne
import matplotlib.pyplot as plt

directory = "./"

train_data = pd.read_csv(directory+'train.csv')
test_data = pd.read_csv(directory+'test.csv')

train_x = train_data.drop(['id', 'label'], axis = 1).values
train_y = train_data['label'].values
test_x = test_data.drop(['id', 'label'], axis=1).values
test_y = test_data['label'].values

train_x = np.where(train_x >= 127, 1, 0)
test_x = np.where(test_x >= 127, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


h_ns = [200]
lrs = [0.0000001]
ks = [1, 20]
for k in ks:
    for h_n in h_ns:
        for learning_rate in lrs:
            print(h_n, learning_rate, k)
            v_m = 784

            W = np.random.randn(h_n, v_m)
            b = np.random.randn(v_m)-2
            c = np.random.randn(h_n)-2
            norms = []

            for x in train_x:

                v_t = np.copy(x)
                h_t = np.zeros(h_n)

                for _ in range(k):
                    pro_samples = np.random.rand(h_n)
                    probs = sigmoid(np.matmul(W, v_t) + c)
                    h_t = np.where(probs < pro_samples, 1, 0)

                    pro_samples = np.random.rand(v_m)
                    probs = sigmoid(np.matmul(h_t.T, W) + b)
                    v_t = np.where(probs < pro_samples, 1, 0)

                update_c_by_vd = sigmoid(np.matmul(W, x) + c)
                update_c_by_vt = sigmoid(np.matmul(W, v_t) + c)
                update_W_by_vd = np.outer(update_c_by_vd, x)
                update_W_by_vt = np.outer(update_c_by_vt, v_t)

                norms.append(np.linalg.norm(update_W_by_vd - update_W_by_vt)+                              np.linalg.norm(x - v_t) +                              np.linalg.norm(update_c_by_vd - update_c_by_vt))

                W = W + learning_rate * (update_W_by_vd - update_W_by_vt)
                b = b + learning_rate * (x - v_t)
                c = c + learning_rate * (update_c_by_vd - update_c_by_vt)

            print("Done")
            plt.plot(norms)
            plt.show()
            plt.clf()

            probs = sigmoid(np.matmul(test_x, W.T) + c)
            pro_samples = np.random.rand(test_x.shape[0], h_n)

            hidden_rep = np.where(probs < pro_samples, 1, 0)
            hidden_rep = probs

            print(hidden_rep.shape)
            print(hidden_rep[np.random.randint(0, test_x.shape[0])])

            Y= tsne(hidden_rep, 2, h_n, 20.0)
            fig, ax = plt.subplots(figsize=(8,8))
            for g in np.unique(test_y):
                i = np.where(test_y == g)
                ax.scatter(Y[i,0], Y[i,1], label=int(g))
            plt.title("t-SNE on "+str(h_n)+"-dimensional hidden representation")
            plt.legend()
            plt.savefig(directory+"clusters_"+str(h_n)+"_"+str(learning_rate).replace('.','')+"_"+str(k)+".png")
            plt.show()
            plt.clf()
