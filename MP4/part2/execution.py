from util import *
from perceptron import *
import numpy as np
import time
import pandas as pd
import seaborn as sb
import differentiable_perceptron


def main():
    start = time.clock()
    classifier = perceptron()
    # classifier = differentiable_perceptron.diff_perceptron()

    trainingData = readDataFile("digitdata/trainingimages", 5000)
    trainingLabel = readLabelFile("digitdata/traininglabels", 5000)

    testData = readDataFile("digitdata/testimages", 1000)
    testLabel = readLabelFile("digitdata/testlabels", 1000)
    # change the following settings like number of epochs,
    # the ordering of training(random vs natural order), etc.
    classifier.setEpoch(100)
    classifier.setDecay(1000)
    classifier.setRandom(True)
    classifier.setRandomize(True)

    # ready for training
    print("Ready for training.")
    classifier.train(trainingData, trainingLabel)
    print("Training is done.")
    accuracy, confusion_matrix = classifier.classify(testData, testLabel)
    print("classification is done.")
    print("-----------------------------------------------------------")
    print("The accuracy is: ", accuracy)
    print("Execution time is: %fs" % (time.clock() - start))
    print(confusion_matrix)
    classifier.training_curve_plot()
    # the following two commands are used to plot something
    # cm_plot(confusion_matrix)
    # plot(classifier)


def plot(classifier):
    # plot the weight of the perceptrons

    for label in range(10):

        temp_array = classifier.perceptrons[label]
        temp_array = temp_array[:28 * 28]
        temp_matrix = np.reshape(temp_array, (28, 28))
        plt.imshow(temp_matrix, interpolation="nearest", cmap='bwr')

        if label == 0:
            plt.clim(vmin=-45, vmax=45)
            plt.colorbar()

        plt.savefig('IMG/perceptron_' + str(label) + '.png')


def cm_plot(matrix):
    new_matrix = np.zeros((10, 10))
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            temp = float(matrix[i][j]) / sum(matrix[i])
            new_matrix[i][j] = round(temp, 2)

    # new_matrix = np.array(matrix)

    df_cm = pd.DataFrame(new_matrix, index=[i for i in [str(x) for x in range(10)]],
                         columns=[i for i in [str(x) for x in range(10)]])
    fig, ax = plt.subplots(figsize=(10, 7))  # Sample figsize in inches
    sb.heatmap(df_cm, annot=True, linewidths=.5, ax=ax)
    # sb.heatmap(df_cm, annot=True)
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
