# stft_layer.py
import tensorflow as tf


class STFTLayer(tf.keras.layers.Layer):
    """
    Custom layer that applies Short-Time Fourier Transform (STFT) to input signals.

    This layer transforms time-domain signals into time-frequency representations
    by computing the magnitude of the STFT.

    Attributes:
        frame_length (int): Length of each frame in samples.
        frame_step (int): Step size between frames in samples.
    """

    def __init__(self, frame_length=64, frame_step=32, **kwargs):
        """
        Initialize the STFT layer.

        Args:
            frame_length (int): Length of each frame in samples. Default is 64.
            frame_step (int): Step size between frames in samples. Default is 32.
            **kwargs: Additional arguments for the parent Layer class.
        """
        super(STFTLayer, self).__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step

    def call(self, inputs):
        """
        Apply STFT to the input tensor.

        Args:
            inputs: Input tensor of shape (batch_size, 1, time_steps)

        Returns:
            Tensor containing the magnitude spectrogram of the STFT.
        """
        # Remove the extra dimension
        # x = tf.squeeze(inputs, axis=1)  # shape: (batch, time)

        # Compute STFT
        stft = tf.signal.stft(
            inputs,
            frame_length=self.frame_length,
            frame_step=self.frame_step
        )

        # Compute magnitude spectrogram
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_config(self):
        """
        Return the configuration of the layer for serialization.

        Returns:
            Dictionary containing the layer configuration.
        """
        config = super(STFTLayer, self).get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step
        })
        return config