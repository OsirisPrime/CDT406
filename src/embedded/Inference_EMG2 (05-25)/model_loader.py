import tflite_runtime.interpreter as tflite
import time

class Model:
    def __init__(self, model_path, logger=None):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.logger = logger

    def get_output_state(self, input_data):
        # input_data must be np.float32, shaped like model input (1, 200, 1)
        input_index = self.input_details[0]['index']
        input_dtype = self.input_details[0]['dtype']
        input_data = input_data.astype(input_dtype)
        self.interpreter.set_tensor(input_index, input_data)
        start = time.time()
        self.interpreter.invoke()
        elapsed = time.time() - start
        output_index = self.output_details[0]['index']
        output_data = self.interpreter.get_tensor(output_index)



        if self.logger:
            print(f"Input Data: {input_data}")
            self.logger.log_output_data(output_data, elapsed)  # ✅ Correct function call

        return output_data
