#!/usr/bin/env python3
"""Time single-image TensorFlow inference on this machine, for comparison with ./build/ml.

Usage (in the Lab 1 virtual environment): python3 scripts/tf_inference_time.py <CNN_TinyImageNet.h5> [data_dir]

Uses the same three test images as the C++ framework (data/image_N.bin, float32 in [0, 1]) and runs on the CPU.
Two timings are reported per image, each as the median of 20 runs after a warm-up:
  predict : one model.predict() call, the end-to-end time measured in Lab 1
  call    : one direct model(x) call, which leaves out most of the per-call overhead of predict()
"""
import os
import platform
import sys
import time

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')  # CPU only, like the C++ implementation

RUNS = 20


def median_ms(fn):
    samples = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples))


def main(model_path, data_dir):
    cpu = next((line.split(':', 1)[1].strip() for line in open('/proc/cpuinfo') if line.startswith('model name')), 'unknown')
    print(f'host: {platform.node()}')
    print(f'Model name: {cpu}')
    print(f'TensorFlow {tf.__version__}, {RUNS} runs per timing, CPU only')

    model = tf.keras.models.load_model(model_path, compile=False)

    print(f'{"image":<7}{"predict (ms)":>14}{"call (ms)":>12}{"class":>8}{"confidence":>12}{"max diff vs reference":>24}')
    for n in range(3):
        image = np.fromfile(os.path.join(data_dir, f'image_{n}.bin'), dtype=np.float32).reshape(1, 64, 64, 3)
        x = tf.constant(image)

        # Warm-up: the first call of each kind builds its graph
        output = model.predict(x, verbose=0)
        model(x, training=False)

        predict_ms = median_ms(lambda: model.predict(x, verbose=0))
        call_ms = median_ms(lambda: model(x, training=False))

        # The exported Lab 1 output of the last layer, as a check that this is the same model and input
        reference = np.fromfile(os.path.join(data_dir, f'image_{n}_data', 'layer_11_output.bin'), dtype=np.float32)
        best = int(np.argmax(output[0]))
        print(f'{n:<7}{predict_ms:>14.2f}{call_ms:>12.2f}{best:>8}{output[0][best]:>12.6f}{np.abs(output[0] - reference).max():>24.3g}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'data')
