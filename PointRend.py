import numpy as np
import matplotlib.pyplot as plt


def _if_near(point, mask, nearest_neighbor):
    nn = nearest_neighbor
    w, h = mask.shape[0], mask.shape[1]
    x, y = point
    mask = np.pad(mask, nn, 'edge')
    x += nn
    y += nn
    if(w+nn > x and h+nn > y):
        x_i, y_i = int(x+0.5), int(y+0.5)
        # return True
        near = mask[x_i-nn:x_i+nn, y_i-nn:y_i+nn]
        if near.max()-near.min() != 0:
            if(x < w and y < h):
                return True
    return False


def getpoint(mask_img,
             k,
             beta,
             training=True,
             nearest_neighbor=3,
             inference_threshold=0.9):
    if(beta > 1 or beta < 0):
        print("fuck")
        raise NameError("beta should be in range [0,1]")
    if(k < 0.1):
        raise NameError("k should be in range [0.1,infinite]")
    w, h = mask_img.shape[0], mask_img.shape[1]
    N = int(beta*k*w*h)
    xy_min = [0, 0]
    xy_max = [w, h]
    points = np.random.uniform(low=xy_min, high=xy_max, size=(N, 2))
    # for the training, the mask is a hard mask
    if training:
        if beta == 0:
            return points
        res = []
        for p in points:
            if(_if_near(p, mask_img, nearest_neighbor)):
                res.append(p)
        others = int((1-beta)*k*w*h)
        not_edge_points = np.random.uniform(low=xy_min,
                                            high=xy_max,
                                            size=(others, 2))
        for p in not_edge_points:
            res.append(p)
        return res
    # for the inference, the mask is a soft mask
    if not training:
        res = []
        for i in range(w):
            for j in range(h):
                if mask_img[i, j] < inference_threshold:
                    res.append((i, j))
        return res


def _generate_mask(size, func=lambda x: (x-7)*(x-7)):
    w, h = size
    res = np.zeros((w, h))
    for x in range(w):
        for y in range(h):
            if y > func(x):
                res[x, y] = 255
    return res


my_mask = _generate_mask((14, 14), )
plt.imshow(my_mask)
points = getpoint(mask_img=my_mask, k=1, beta=1, nearest_neighbor=2)
print(points)
points = list(zip(*points))
plt.scatter(points[1], points[0], c='black', s=4)
plt.show()
