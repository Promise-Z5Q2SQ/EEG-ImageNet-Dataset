from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def SVM_classifier(X_train, y_train, X_test, y_test):
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def RF_classifier(X_train, y_train, X_test, y_test):
    dt_model = RandomForestClassifier(n_estimators=200)
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def KNN_classifier(X_train, y_train, X_test, y_test):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def DT_classifier(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred = dt_model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def Ridge_classifier(X_train, y_train, X_test, y_test):
    ridge_model = RidgeClassifier()
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def MLP_classifier(X_train, y_train, X_test, y_test):
    mlp_model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(256, 128, 64), activation='relu',
                              learning_rate='adaptive', )
    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test)
    return accuracy_score(y_test, y_pred)
