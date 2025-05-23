import numpy as np
from collections import deque
from preprocessor import SignalPreprocessor

#  Read -> Windowing -> Filter -> Feature Extraction -> Model -> Output
pre_processor_variant = 1
sample_rate = 1000

preprocessor = SignalPreprocessor(pre_processor_variant=pre_processor_variant,
                                               low_freq=20.0,
                                               high_freq=499.0,
                                               fs=sample_rate,
                                               order=7,
                                               down_sample=False,
                                               )

class DataProcess:
    def __init__(self, config, data_input, logger=None):
        self.config = config
        self.data_input = data_input
        self.logger = logger

        self.buffer = deque(maxlen=config.buffer_len)
        self.step = int(config.read_window_size * config.window_overlap)
        self.preprocessor = SignalPreprocessor(
            low_freq=config.low_cut,
            high_freq=config.high_cut,
            fs=config.sampling_rate,
            order=config.filter_order,
            down_sample=False  # Already downsampled if needed
        )

    def _get_next_window(self):
        # Fill buffer until full or no more data
        while len(self.buffer) < self.config.buffer_len:
            if self.data_input.has_next():
                sample = self.data_input.next()
                if self.logger:
                    self.logger(sample)
                self.buffer.append(sample)
            else:
                # No more data
                return None

        # Flatten buffer and take last 200 samples (window size)
        window_data = np.array(self.buffer, dtype=np.float32).flatten()[-self.config.read_window_size:]
        # Preprocess window: filter, normalize etc.
        processed = self.preprocessor.pre_process(window_data)
        return processed

    def get_next(self):
        window = self._get_next_window()
        if window is None:
            return None
        # Reshape to (1, 200, 1)
        window = window.reshape(1, self.config.read_window_size, 1) # Change to (1, self.config.read_window_size) for all other than LSTM
        return window.astype(np.float32)
