import time
import numpy as np
import Adafruit_BBIO.ADC as ADC  # For EMG or sensor input
import tflite_runtime.interpreter as tflite

# Initialize ADC
ADC.setup()

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input/output tensor info
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def read_sensor():
    # Read from EMG sensor connected to P9_40 (example pin)
    value = ADC.read("P9_40")  # Range 0.0 to 1.0
    return value

def preprocess(value):
    # Example preprocessing (normalize and reshape)
    input_shape = input_details[0]['shape']
    input_data = np.array([[value]], dtype=np.float32)  # Adjust shape/type
    return input_data

def postprocess(output_data):
    # Interpret model output
    predicted_class = np.argmax(output_data)
    return predicted_class

# Real-time loop
try:
    while True:
        raw_value = read_sensor()
        input_data = preprocess(raw_value)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Postprocess and act
        result = postprocess(output_data)
        print(f"Prediction: {result}")

        # Optionally control GPIO based on result
        # if result == 1:
        #     activate_motor()

        time.sleep(0.05)  # 20 Hz loop

except KeyboardInterrupt:
    print("Real-time inference stopped.")
