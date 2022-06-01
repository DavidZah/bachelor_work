import matplotlib.pyplot as plt
import numpy as np  # linear algebra



networks = ["Resnet34\n Rozlišení 512 x 512\n,BatchSize = 8",
            "Resnet34\n Rozlišení 800 x 800\n,BatchSize = 2",
            "Resnet101\n Rozlišení 800 x 800\n,BatchSize = 2",
            "Resnet34\n Rozlišení 1024 x 1024\n,BatchSize = 1"
            ]


lst = [
np.load("data/plots/lst_(512, 512)_resnet34_batch_8.npy"),
np.load("data/plots/lst_(800, 800)_resnet34_batch_2.npy"),
np.load("data/plots/lst_(800, 800)_resnet101_batch_1.npy"),
np.load("data/plots/lst_(1024, 1024)_resnet34_batch_1.npy")
]

not_fount_points = []

median_lst = []
for i in lst:
    indexes = np.where(np.array(i) == -1)[0]
    oper_lst = []
    for j in i:
        if j != -1:
            oper_lst.append(j)
    median_lst.append(oper_lst)

for index,i in enumerate(median_lst):
    print(networks[index])
    print(np.mean(i))
plt.boxplot(lst[0])
plt.show()
print("done")