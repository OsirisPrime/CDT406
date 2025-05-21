from pathlib import Path
import numpy as np
import struct
import array
import time
import sys

from tflite_interpreter import TFLiteModelInterpreter
from preprocessor import SignalPreprocessor

v_ref = 1.8
power = (1 << 12) - 1

def value_to_voltage(val):
    return val * v_ref / power

times = []
inference_times = []

def exit_gracefully():
    log = open('/home/EMG3/setup/model.log', 'w')
    log.write('Waiting\n')
    log.write(str(times))
    log.write('\nInference\n')
    log.write(str(inference_times))
    log.close()
    print(f"Average: {sum(times)/len(times)}, {sum(inference_times)/len(inference_times)}")
    adc_in.close()
    inference_out.close()
    exit(0)

# === CONFIGURATION ===
model_path = Path(sys.argv[1])
pre_processor_variant = 1
sample_rate = 1000
window_size = 200
step_size = 50

preprocessor = SignalPreprocessor(pre_processor_variant=pre_processor_variant,
                                               low_freq=20.0,
                                               high_freq=500.0,
                                               fs=sample_rate,
                                               order=7,
                                               down_sample=False,
                                               )

model = TFLiteModelInterpreter(str(model_path))
print('TFLite model is loaded')

buffer_size = 1500
number_of_states = 4

in_struct_fmt = f'<{buffer_size * 2}B'
out_struct_fmt = f'<{buffer_size * 2}B{number_of_states}f'
in_pipe = '/home/EMG3/setup/adc_pipe'
out_pipe = '/home/EMG3/setup/inference_pipe'

adc_in = open(in_pipe, "rb")
inference_out = open(out_pipe, "wb")
print('Starting')


try:
    while True:
        now = time.perf_counter()
        received = adc_in.read(struct.calcsize(in_struct_fmt))

        if len(received) > 0:
            start = time.perf_counter()
            data = array.array('H')
            data.frombytes(received)
            del data[2::3]
            voltages = list(map(value_to_voltage, data))

            try:
                processed = preprocessor.pre_process(voltages)
                model_input = processed.astype(np.float32)
            except Exception as e:
                print(f"Preprocessing failed: {e}")
                break

            res = model.predict(model_input)
            end = time.perf_counter()
            now = end - now
            start = end - start
            packed = struct.pack(out_struct_fmt, *received, *res)

            try:
                inference_out.write(packed)
                inference_out.flush()
            except IOError:
                log = open('/home/EMG3/setup/model.log', 'w')
                log.write('Waiting\n')
                log.write(str(times))
                log.write('\nInference\n')
                log.write(str(inference_times))
                log.close()
                print(f"Average: {sum(times)/len(times)}, {sum(inference_times)/len(inference_times)}")
                adc_in.close()
                exit(0)

            times.append(now)
            inference_times.append(start)
        else:
            break
except KeyboardInterrupt:
    exit_gracefully()

exit_gracefully()