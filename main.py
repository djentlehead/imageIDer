import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

(trainingImages, trainingLabels), (testingImages, testingLabels) = datasets.cifar10.load_data()
trainingImages, testingImages = trainingImages / 255, testingImages / 255

classNames = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trainingImages[i], cmap = plt.cm.binary)
    plt.xlabel(classNames[trainingLabels[i][0]])


plt.show()

# trainingImages = trainingImages[:20000]
# trainingLabels = trainingLabels[:20000]
#
# testingImages = testingImages[:4000]
# testingLabels = testingLabels[:4000]


model = models.load_model("image_classifier.model")

img = cv.imread(r"C:\Users\Bhav\PycharmProjects\imageRecog\image_classifier.model\deer.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img, cmap = plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)
index = np.argmax(prediction)
print(f"Prediction is {classNames[index]}")

plt.show()




