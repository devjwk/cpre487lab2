#!/usr/bin/env python3
"""Copy the Lab 1 exports into data/ in the layout the framework reads.

Usage: python3 scripts/import_lab1_data.py <lab1_binaries_dir> [data_dir]

<lab1_binaries_dir> holds img_data/ and model_data/ as exported by the Lab 1 notebook
(for example cpre487lab1/submission/lab1_binaries_06).

  img_data/image_N.bin (uint8, 64x64x3)             -> data/image_N.bin (float32, value / 255)
  img_data/test_input_N/<keras layer>_output.bin    -> data/image_N_data/layer_<i>_output.bin
  model_data/<keras layer>_weights.bin / _bias.bin  -> data/model/<name>_weights.bin / _biases.bin
"""
import array
import pathlib
import shutil
import sys

# Keras layer names in model order; the index is the framework's layer number
LAYERS = ['conv2d', 'conv2d_1', 'max_pooling2d', 'conv2d_2', 'conv2d_3', 'max_pooling2d_1',
          'conv2d_4', 'conv2d_5', 'max_pooling2d_2', 'flatten', 'dense', 'dense_1']
# Keras layer name -> weight file prefix used by buildToyModel() in src/ML.cpp
WEIGHT_NAMES = {'conv2d': 'conv1', 'conv2d_1': 'conv2', 'conv2d_2': 'conv3', 'conv2d_3': 'conv4',
                'conv2d_4': 'conv5', 'conv2d_5': 'conv6', 'dense': 'dense1', 'dense_1': 'dense2'}


def main(src, dst):
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    (dst / 'model').mkdir(parents=True, exist_ok=True)

    for n in range(3):
        pixels = (src / 'img_data' / f'image_{n}.bin').read_bytes()
        assert len(pixels) == 64 * 64 * 3, f'image_{n}.bin: expected 12288 uint8 values'
        # Same preprocessing as the Lab 1 notebook: uint8 [0, 255] -> float32 [0, 1]
        (dst / f'image_{n}.bin').write_bytes(array.array('f', (p / 255 for p in pixels)).tobytes())

        out_dir = dst / f'image_{n}_data'
        out_dir.mkdir(exist_ok=True)
        for i, layer in enumerate(LAYERS):
            shutil.copyfile(src / 'img_data' / f'test_input_{n}' / f'{layer}_output.bin',
                            out_dir / f'layer_{i}_output.bin')

    for layer, name in WEIGHT_NAMES.items():
        shutil.copyfile(src / 'model_data' / f'{layer}_weights.bin', dst / 'model' / f'{name}_weights.bin')
        shutil.copyfile(src / 'model_data' / f'{layer}_bias.bin', dst / 'model' / f'{name}_biases.bin')

    print(f'Imported 3 images, {3 * len(LAYERS)} layer outputs and {2 * len(WEIGHT_NAMES)} weight files into {dst}/')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'data')
