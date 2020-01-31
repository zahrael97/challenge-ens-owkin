import matplotlib.pyplot as plt
import numpy as np

test = np.load('data/x_test/images/patient_005.npz', 'r')
print(test['scan'].shape, test['mask'].shape)

colored_images = np.zeros(list(test['scan'].shape) + [3])
M_scan = max(-np.min(test['scan']), np.max(test['scan']))
colored_images[:, :, :, 2] += test['scan']
colored_images[:, :, :, 0] += test['mask']

for i in range(30):
    plt.imshow(colored_images[:, :, i * 2], cmap = 'gray')
    plt.draw()
    plt.waitforbuttonpress()
    plt.clf()
