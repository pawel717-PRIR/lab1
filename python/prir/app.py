from prir.config import app_logging
from prir.machinelearning import dataset
from prir.metrics import timer
import prir.machinelearning.standarization.standarizator as standarizator

logger = app_logging.get_logger(__name__)
logger.debug("APPLICATION START")

train_data = dataset.read_test_data()
standarized_min_max = timer.call_and_measure("""standarizator.min_max_scaler(train_data)""", number=2)
standarized_standard_scaler = timer.call_and_measure("""standarizator.standard_scaler(train_data)""", number=2)

logger.debug("APPLICATION END")





