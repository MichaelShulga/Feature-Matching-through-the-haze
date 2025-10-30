import numpy as np
import cv2

def make_perspective_transform(m):

    def transform(x):
        points = np.array(x, dtype=np.float32)
        return cv2.perspectiveTransform(points.reshape(-1, 1, 2), m).reshape(-1, 2)
    
    return transform

def make_loss(p):

    def loss(vestigial, region):
        d = lambda x: np.linalg.norm(vestigial(x) - x, axis=1)
        if p == np.inf:
            return d(region).max()
        return ((d(region) ** p).sum() / len(region)) ** (1/p)
    
    return loss

loss_2 = make_loss(2)
loss_inf = make_loss(np.inf)
