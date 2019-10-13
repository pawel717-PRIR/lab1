from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from prir.config import app_logging


def train_predict(train_data, train_target, test_data, test_target):
    model = KNeighborsClassifier(n_neighbors=1, algorithm='auto')

    # Train the model using the training set
    model.fit(train_data, train_target)

    # Predict Output
    predicted = model.predict(test_data)
    logger.info("KNN Classifier stats: \n" + classification_report(test_target, predicted))

    return predicted


logger = app_logging.get_logger(__name__)
