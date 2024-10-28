from keras.datasets import mnist
from matplotlib import pyplot


def load_and_display_mnist():
    
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
 

    for i in range(9):  
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    

    return (train_X, train_y), (test_X, test_y)

(train_X, train_y), (test_X, test_y) = load_and_display_mnist()