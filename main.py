import numpy as np
import math
import random
from matplotlib import pyplot as plt
import json

def activation_func(x):
    return  round((math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x)), 5)


def train(mat1,mat2,n,m):
    for i in range(0,n*m):
        for j in range(0,n*m):
            if i != j:
                mat2[i][j] = mat2[i][j] + mat1[0][i] * mat1[0][j]
            else:
                mat2[i][j] = 0
    return mat2

def show_image(initial_image, corrupted_image, restored_image):
    fig, axs = plt.subplots(3, figsize=(5, 15))
    axs[0].imshow(initial_image, cmap='Greys_r')
    axs[1].imshow(corrupted_image, cmap='Greys_r')
    axs[2].imshow(restored_image, cmap='Greys_r')
    plt.show()

def create_data(X,Y,W):
    Flag = True
    while Flag:
        S = 0
        nr = random.randrange(0, n * m, 1)
        for i in range(0, n * m):
            S = S + Y[0][i] * W[i][nr]
        sign = activation_func(S)
        if sign != Y[0][nr]:
            Y[0][nr] = sign
        if np.array_equal(X,Y):
            Flag = False
    return Y

with open('image.txt', 'r') as rd1:
    images = json.load(rd1)
with open('corrupted.txt', 'r') as rd2:
    corrupted_images = json.load(rd2)

X=images
Y=corrupted_images

n = len(images)
m = len(images[0])

X=np.array(X).reshape((1,n*m))
Y=np.array(Y).reshape((1,n*m))

W = np.zeros((n*m,n*m),dtype=np.int16)
W = train(X,W,n,m)
Y=create_data(X,Y,W)

Y = Y.reshape((n, m))
show_image(images,corrupted_images,Y)