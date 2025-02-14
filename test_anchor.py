import numpy as np
from retinanet.anchors import Anchors

def test_compute_shape():
    a = Anchors(sizes=[8,16,32,64,128])
    # a = Anchors()

    image = np.random.rand(8,3,1024,1024)

    anchors = a(image)

    print(anchors.shape)
    print (anchors)


if __name__ == '__main__':
    test_compute_shape()