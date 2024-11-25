from sklearn import model_selection
from sklearn.svm import SVR

def def_and_train_ml(data):
    X = data.drop('RUL', axis=1)
    X = X.drop('unit_number', axis=1, errors='ignore')
    X = X.drop('time_cycle_unit', axis=1, errors='ignore')
    y = data['RUL']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

    # model definition
    svr = SVR(kernel='rbf')
        
    # model training
    svr.fit(X_train, y_train)

    return svr, X_test, y_test, "sklearn", {"kernel": "rbf"}, {"type": "SVR", "params": svr.get_params()}, 1