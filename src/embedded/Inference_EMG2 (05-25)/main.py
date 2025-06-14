import numpy as np
from led_control import LedControl
from analog_input import FileInput, SensorInput
from data_process import DataProcess
from logger import Logger
from config import Config
import time
from model_loader import Model

def get_data_input(type='File'):
    if type == 'File':
        return FileInput(
            file_name=config.file_path,
            sampling_rate=config.sampling_rate,
            window_size=config.read_window_size
        )
    elif type == 'Sensor':
        return SensorInput(
            sampling_rate=config.sampling_rate,
            window_size=config.read_window_size
        )
    else:
        raise ValueError("Invalid input type")

if __name__ == '__main__':
    config = Config('preprocessing_config.toml')
    logger = Logger(config.log_path)
    led_control = LedControl()
    data_input = get_data_input('Sensor')
    data_process = DataProcess(config=config, data_input=data_input, logger=logger.log_input_data)
    model = Model(model_path=config.model_path, logger=logger)

    while True:
        start_time = time.time()

        window = data_process.get_next()
        if window is None:
          print("Waiting for new data...")
          time.sleep(0.02)  # Give time for sensor data to arrive
          continue
        preprocessing_time = time.time()

        # window shape should be (1, 200, 1)
        output_state = model.get_output_state(window)  # Original input processing
        print(f"Model Output: {np.round(output_state * 100, 2)}%")

        # Test with random input
        #test_input = np.random.uniform(0.0, 1.8, (1, 200, 1)).astype(np.float32) # Change to (1, 200) for all other than LSTM
        #test_output = model.get_output_state(test_input)
        #print(f"Test Output: {np.round(test_output * 100, 2)}%")

        end_time = time.time()

        total_inference_time = end_time - start_time
        pre_preprocess_time = preprocessing_time - start_time
        prediction_time = end_time - preprocessing_time

        print("Total inference time:", total_inference_time)
        print("\nPreprocessing time:", pre_preprocess_time)
        print("\nPrediction time:", prediction_time)
        print("\nModel output:", output_state)
        led_control.set_state(output_state)
