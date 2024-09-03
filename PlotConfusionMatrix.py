import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class PlotConfusionMatrix:
    def plot_confusion_matrix(self, labels, cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def prepareWork(self, labels, y_true, y_pred):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        tick_marks = np.array(range(len(labels))) + 0.5
        cm = confusion_matrix(y_true, y_pred)
        np.set_printoptions(precision=2)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 8), dpi=120)

        ind_array = np.arange(len(labels))
        x, y = np.meshgrid(ind_array, ind_array)

        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm_normalized[y_val][x_val]
            if c > 0.01:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
        # offset the tick
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)

        self.plot_confusion_matrix(labels, cm_normalized, title='Normalized confusion matrix')
        # show confusion matrix
        # plt.savefig('image/confusion_matrix.png', format='png')
        plt.show()


# Plotting the confusion matrix
# attacks is the list of labels
# y_test is the test result
# y_pred is the prediction result
def plotMatrix(attacks, y_test, y_pred):
    def filter_invalid_chars(text):
        return ''.join(c for c in text if c.isprintable())
    # When dividing the test set, the labels with small number are eliminated
    y_test_set = set(y_test)
    y_test_list = list(y_test_set)
    attacks_test = []
    for i in range(0, len(y_test_set)):
        filtered_label = filter_invalid_chars(attacks[y_test_list[i]])
        attacks_test.append(filtered_label)

    p = PlotConfusionMatrix()
    p.prepareWork(attacks_test, y_test, y_pred)
