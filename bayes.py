from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np

PLOT_SIZE = 100

BAYES_RADIUS = 0.7


data = []
for i in range(50):
    #Red data set
    data.append((np.random.randn() - 1, np.random.randn() - 1, 0))
    #Blue data set
    data.append((np.random.randn() + 1, np.random.randn() + 1, 1))

# Point to classify
point = (np.random.randn(), np.random.randn())


#Transforms a number into a color
def getColor(n=0.5):
    if(n is 0.5):
        return 'green'
    elif(n < 0.5):
        return 'red'
    else:
        return 'blue'

#Tests to see if the testCenter is within range of the circleCenter
def inBayesRange(circleCenter, testCenter):
    circlePoint = (circleCenter[0], circleCenter[1])
    testPoint = (testCenter[0], testCenter[1])
    return distance.euclidean(circlePoint, testPoint) <= BAYES_RADIUS

#Gets a numerical prediction based on the circleCenter location
def getPrediction(circleCenter):
    circlePoint = (circleCenter[0], circleCenter[1])
    inRange = []
    for dataVal in data:
        if(inBayesRange(circlePoint, dataVal)):
            inRange.append(dataVal)

    total = 0
    if(len(inRange) > 0):
        for inRangeVal in inRange:
            total += inRangeVal[2]

        return total / len(inRange)
    else:
        return 0.5


#Gets a prediction color for a given circle center
def getPredictionColor(circleCenter):
    prediction = getPrediction(circleCenter)
    return getColor(prediction)


axes = plt.gca()
axes.set_xlim([-4, 4])
axes.set_ylim([-4, 4])

for trips in data:
    # Plot in radius points
    if(inBayesRange(point, trips)):
        plt.scatter(trips[0], trips[1], s=PLOT_SIZE,
                    c=getColor(trips[2]), alpha=0.8)
    # Plot out of radius points
    else:
        plt.scatter(trips[0], trips[1], s=PLOT_SIZE,
                    c=getColor(trips[2]), alpha=0.1)
# Plot point in question
plt.scatter(point[0], point[1], s=PLOT_SIZE,
            c=getPredictionColor(point), alpha=1.0)
plt.scatter(point[0], point[1], s=PLOT_SIZE * 6, c='gray', alpha=0.3)

print("Prediction: {}".format(getPrediction(point)))
print("Prediction Color: {}".format(getPredictionColor(point)))

plt.show()