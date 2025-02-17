from Model.Anchors import Anchors
import numpy as np

a = Anchors(pyramid_levels=[2,3,4,5,6])

image = np.random.rand(4, 3, 1024, 1024)

print(a(image).shape)

reg = 