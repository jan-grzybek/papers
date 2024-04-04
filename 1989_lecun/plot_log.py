import math
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

epochs = 100

lr_results = {"lr": []}
with open("log.csv", "r") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        lr = -math.log(10 * float(row[0]), 2)
        if lr not in lr_results.keys():
            lr_results["lr"].append(lr)
            lr_results[lr] = []
        lr_results[lr].append([float(datapoint) for datapoint in row[2:]])

min_test_error = 100
lr_minimizing = None
for lr in lr_results["lr"]:
    assert len(lr_results[lr]) == epochs
    if lr_results[lr][22][3] < min_test_error:
        min_test_error = lr_results[lr][22][3]
        lr_minimizing = lr

print(f"Best result after 23 epochs achieved at learning rate = {2**(-lr_minimizing)/10}:")
print(f"training error: {round(lr_results[lr_minimizing][22][1], 2)}%, test error: {round(min_test_error, 2)}%")
print(f"training loss: {lr_results[lr_minimizing][22][0]}, test loss: {lr_results[lr_minimizing][22][2]}")


def polygon_under_graph(x, y):
    """
    Construct the vertex list which defines the polygon filling the space under
    the (x, y) line graph. This assumes x is in ascending order.
    """
    return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]


lrs = lr_results["lr"]

for index, label in [(0, 'training loss [MSE]'), (2, 'test loss [MSE]'),
                     (1, 'training error [%]'), (3, 'test error [%]')]:
    ax = plt.figure().add_subplot(projection='3d')

    x = {lr: [i + 1 for i in range(len(lr_results[lr]))] for lr in lrs}
    y = {lr: [row[index] for row in lr_results[lr]] for lr in lrs}
    max_loss = 0
    for lr in lrs:
        max_loss = max(max_loss, max(y[lr]))

    gamma = np.vectorize(math.gamma)
    verts = [polygon_under_graph(x[lr], y[lr]) for lr in lrs]
    facecolors = plt.colormaps['viridis_r'](np.linspace(0, 1, len(verts)))

    poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
    ax.add_collection3d(poly, zs=lrs, zdir='y')
    ax.set(xlim=(1, epochs), ylim=(min(lrs), max(lrs)+0.1), zlim=(0, max_loss),
           xlabel='epochs', ylabel='learning rate [2^(-x)/10]', zlabel=label)

    plt.show()
