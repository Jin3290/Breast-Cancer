# main.py
from sklearn import metrics
import pandas
from sklearn.datasets import load_breast_cancer
from Modelli.Classificazione import ClassificazioneModel

if __name__ == "__main__":
    dataset = load_breast_cancer()
    dir(dataset)

    dataframe = pandas.DataFrame(dataset.data,
                                 columns=dataset.feature_names)

    SVM_model = ClassificazioneModel()
    X_train, X_test, y_train, y_test = SVM_model.load_and_split_data(dataset)
    
    print("Classificazione con il modello SVM, risultati:")
    model = SVM_model.get_model()
    prediction = model.predict(X_test)
    print(metrics.recall_score(y_test, prediction))
    print(metrics.precision_score(y_test, prediction))
    print(metrics.accuracy_score(y_test, prediction))