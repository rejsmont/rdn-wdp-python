#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

plt.figure()

mu = 0
variance2 = 1
sigma2 = math.sqrt(variance2)
x2 = np.linspace(mu - 15*sigma2, mu + 15*sigma2, 100)
plt.plot(x2,mlab.normpdf(x2, mu, sigma2))
variance1 = 5
sigma1 = math.sqrt(variance1)
x1 = np.linspace(mu - 5*sigma1, mu + 5*sigma1, 100)
plt.plot(x2,mlab.normpdf(x1, mu, sigma1))
plt.savefig('/Users/radoslaw.ejsmont/Desktop/gaussians.pdf')
plt.show()

plt.figure()
plt.plot(x2,mlab.normpdf(x2, mu, sigma2) - mlab.normpdf(x1, mu, sigma1))
plt.savefig('/Users/radoslaw.ejsmont/Desktop/gaussians_dog.pdf')
plt.show()
