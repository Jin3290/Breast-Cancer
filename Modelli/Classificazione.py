# Modello di Classificazione
from sklearn import svm
from sklearn.model_selection import train_test_split

class ClassificazioneModel:
    def __init__(self):
        self.model = svm.SVC(kernel="linear")

    def load_and_split_data(self, dataset):
        X_train, X_test, y_train, y_test = train_test_split(dataset.data,
                                                            dataset.target,
                                                            random_state=0)
        self.model.fit(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def get_model(self):
        return self.model