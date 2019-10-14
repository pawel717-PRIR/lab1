from prir.config import app_logging
from prir.files.mnist_file_manager import MnistFileManager
from prir.metrics import timer
import prir.machinelearning.standarization.standarizator as standarizator
import prir.machinelearning.classifier.knn as knn


logger = app_logging.get_logger(__name__)
logger.debug("APPLICATION START")

dataset = MnistFileManager().read_dataset()

train_data_standarized_min_max = timer.call_and_measure(
        """standarizator.min_max_scaler(dataset.get('train_data'))""",
        number=10)
timer.call_and_measure(
        """knn.train_predict(train_data_standarized_min_max, dataset.get('train_target'), 
        dataset.get('test_data'), dataset.get('test_target'))""",
        number=10)
del train_data_standarized_min_max


train_data_standarized_standard_scaler = timer.call_and_measure(
    """standarizator.standard_scaler(dataset.get('train_data'))""",
    number=1)
timer.call_and_measure(
        """knn.train_predict(train_data_standarized_standard_scaler, dataset.get('train_target'), 
        dataset.get('test_data'), dataset.get('test_target'))""",
        number=1)
del train_data_standarized_standard_scaler

logger.debug("APPLICATION END")
