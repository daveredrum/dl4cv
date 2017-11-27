import pickle as pickle
import os

def save_model(modelname, data):
    dir = 'models'
    model = {modelname: data}
    if not os.path.exists(dir):
        os.makedirs(dir)
    pickle.dump(model, open(dir + '/' + modelname + '.p', 'wb'))


def save_softmax_classifier(classifier):
    modelname = 'softmax_classifier'
    save_model(modelname, classifier)


def save_two_layer_net(classifier):
    modelname = 'two_layer_net'
    save_model(modelname, classifier)


def save_feature_neural_net(classifier):
    modelname = 'feature_neural_net'
    save_model(modelname, classifier)