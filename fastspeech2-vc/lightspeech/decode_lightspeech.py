# -*- coding: utf-8 -*-
# Copyright 2020 Minh Nguyen (@dathudeptrai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decode trained FastSpeech from folders."""

import argparse
import logging
import os
import sys

sys.path.append(".")

import numpy as np
import tensorflow as tf
import yaml
from tqdm import tqdm
from tensorflow_tts.utils import find_files
#from examples.fastspeech.fastspeech_dataset import CharactorDataset
#from tensorflow_tts.configs import FastSpeech2Config, LightSpeechConfig
#from tensorflow_tts.models import TFFastSpeech2
from lightspeech import TFLightSpeech,LightSpeechConfig
from tensorflow_tts.datasets.abstract_dataset import AbstractDataset

class CharactorDataset(AbstractDataset):
    """Tensorflow Charactor dataset."""

    def __init__(
        self, root_dir, charactor_query="*-ids.npy", charactor_load_fn=np.load,
    ):
        """Initialize dataset.

        Args:
            root_dir (str): Root directory including dumped files.
            charactor_query (str): Query to find charactor files in root_dir.
            charactor_load_fn (func): Function to load charactor file.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of charactor and mel files.
        charactor_files = sorted(find_files(root_dir, charactor_query))

        # assert the number of files
        assert (
            len(charactor_files) != 0
        ), f"Not found any char or duration files in ${root_dir}."
        if ".npy" in charactor_query:
            suffix = charactor_query[1:]
            utt_ids = [os.path.basename(f).replace(suffix, "") for f in charactor_files]

        # set global params
        self.utt_ids = utt_ids
        self.charactor_files = charactor_files
        self.charactor_load_fn = charactor_load_fn

    def get_args(self):
        return [self.utt_ids]

    def generator(self, utt_ids):
        for i, utt_id in enumerate(utt_ids):
            charactor_file = self.charactor_files[i]
            charactor = self.charactor_load_fn(charactor_file)

            items = {"utt_ids": utt_id, "input_ids": charactor}

            yield items

    def create(
        self,
        allow_cache=False,
        batch_size=1,
        is_shuffle=False,
        map_fn=None,
        reshuffle_each_iteration=True,
    ):
        """Create tf.dataset function."""
        output_types = self.get_output_dtypes()
        datasets = tf.data.Dataset.from_generator(
            self.generator, output_types=output_types, args=(self.get_args())
        )

        if allow_cache:
            datasets = datasets.cache()

        if is_shuffle:
            datasets = datasets.shuffle(
                self.get_len_dataset(),
                reshuffle_each_iteration=reshuffle_each_iteration,
            )

        # define padded shapes
        padded_shapes = {"utt_ids": [], "input_ids": [None]}

        datasets = datasets.padded_batch(
            batch_size, padded_shapes=padded_shapes, drop_remainder=True
        )
        datasets = datasets.prefetch(tf.data.experimental.AUTOTUNE)
        return datasets

    def get_output_dtypes(self):
        output_types = {"utt_ids": tf.string, "input_ids": tf.int32}
        return output_types

    def get_len_dataset(self):
        return len(self.utt_ids)

    def __name__(self):
        return "CharactorDataset"


def main():
    """Run fastspeech2 decoding from folder."""
    parser = argparse.ArgumentParser(
        description="Decode soft-mel features from charactor with trained FastSpeech "
        "(See detail in examples/fastspeech2/decode_fastspeech2.py)."
    )
    parser.add_argument(
        "--rootdir",
        default=None,
        type=str,
        required=True,
        help="directory including ids/durations files.",
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save generated speech."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkpoint file to be loaded."
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        required=True,
        help="yaml format configuration file. if not explicitly provided, "
        "it will be searched in the checkpoint directory. (default=None)",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        required=False,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    args = parser.parse_args()

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # load config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))

    if config["format"] == "npy":
        char_query = "*-ids.npy"
        char_load_fn = np.load
    else:
        raise ValueError("Only npy is supported.")

    # define data-loader
    dataset = CharactorDataset(
        root_dir=args.rootdir,
        charactor_query=char_query,
        charactor_load_fn=char_load_fn,
    )
    dataset = dataset.create(batch_size=args.batch_size)

    # define model and load checkpoint
    lightspeech = TFLightSpeech(
        config=LightSpeechConfig(**config["lightspeech_params"]), name="lightspeech"
    )
    lightspeech._build()
    lightspeech.load_weights(args.checkpoint)

    for data in tqdm(dataset, desc="Decoding"):
        utt_ids = data["utt_ids"]
        char_ids = data["input_ids"]

        # fastspeech inference.
        (
            masked_mel_before,
            masked_mel_after,
            duration_outputs,
            _,
        ) = lightspeech.inference(
            char_ids,
            speaker_ids=tf.zeros(shape=[tf.shape(char_ids)[0]], dtype=tf.int32),
            speed_ratios=tf.ones(shape=[tf.shape(char_ids)[0]], dtype=tf.float32),
            f0_ratios=tf.ones(shape=[tf.shape(char_ids)[0]], dtype=tf.float32),
        )

        # convert to numpy
        masked_mel_befores = masked_mel_before.numpy()
        masked_mel_afters = masked_mel_after.numpy()

        for (utt_id, mel_before, mel_after, durations) in zip(
            utt_ids, masked_mel_befores, masked_mel_afters, duration_outputs
        ):
            # real len of mel predicted
            real_length = durations.numpy().sum()
            utt_id = utt_id.numpy().decode("utf-8")
            # save to folder.
            np.save(
                os.path.join(args.outdir, f"{utt_id}-fs-before-feats.npy"),
                mel_before[:real_length, :].astype(np.float32),
                allow_pickle=False,
            )
            np.save(
                os.path.join(args.outdir, f"{utt_id}-fs-after-feats.npy"),
                mel_after[:real_length, :].astype(np.float32),
                allow_pickle=False,
            )


if __name__ == "__main__":
    main()