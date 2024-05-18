from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier


class SimpleModel:
    def __init__(self, args):
        if args.model.lower() == 'svm':
            model = SVC(kernel='linear', C=1.0, probability=True)
        elif args.model.lower() == 'rf':
            model = RandomForestClassifier(n_estimators=200)
        elif args.model.lower() == 'knn':
            model = KNeighborsClassifier(n_neighbors=5)
        elif args.model.lower() == 'dt':
            model = DecisionTreeClassifier(criterion='entropy')
        elif args.model.lower() == 'ridge':
            model = RidgeClassifier()
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
