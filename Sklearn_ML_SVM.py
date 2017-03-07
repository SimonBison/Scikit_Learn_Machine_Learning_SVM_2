# Predicting digits with SVM

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

clf = svm.SVC(gamma=0.001, C=100)

# shows how many we have examples of digits
print(len(digits.data))

# storing all of the answers
x, y = digits.data[:-10], digits.target[:-10]
clf.fit(x, y)

# predict what is the negative first element
print('Prediction:', clf.predict(digits.data[-3]))

plt.imshow(digits.images[-3], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()