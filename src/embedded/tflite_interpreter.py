import numpy as np
import tflite_runtime.interpreter as tflite

class TFLiteModelInterpreter:
    def __init__(self, model_path: str):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        input_index = self.input_details[0]['index']
        input_dtype = self.input_details[0]['dtype']
        input_data = input_data.astype(input_dtype)
        self.interpreter.set_tensor(input_index, input_data)
        self.interpreter.invoke()
        output_index = self.output_details[0]['index']
        return self.interpreter.get_tensor(output_index)
