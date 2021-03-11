from collections import defaultdict
import datetime
import io
import json
import pathlib
import pickle
import re
import time
import uuid

import numpy as np
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import mlflow


class Logger:

    def __init__(self, logdir, log_mlflow=False, run_name=None, params={}):
        self._logdir = logdir
        self._log_mlflow = log_mlflow
        self._writer = tf.summary.create_file_writer(str(logdir), max_queue=1000)
        self._last_step = None
        self._last_time = None
        self._scalars = {}
        self._scalars_mean = defaultdict(list)
        self._images = {}
        self._videos = {}
        if log_mlflow:
            if run_name:
                run_name = run_name.format(**params)
            mlflow.start_run(run_name=run_name)
            mlflow.log_params(dict(list(params.items())[:100]))  # Max 100 batch
            mlflow.log_params(dict(list(params.items())[100:]))

    def scalar(self, name, value):
        self._scalars[name] = float(value)

    def scalar_mean(self, name, value):
        self._scalars_mean[name].append(float(value))

    def image(self, name, value):
        self._images[name] = np.array(value)

    def video(self, name, value):
        self._videos[name] = np.array(value)

    def write(self, step, fps=False):
        scalars = self._scalars
        for name, values in self._scalars_mean.items():
            scalars[name] = np.mean(values)
        if fps:
            scalars['fps'] = self._compute_fps(step)

        # console
        print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in scalars.items()))

        # json
        with (self._logdir / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps({'step': step, **scalars}) + '\n')

        # tensorboard
        with self._writer.as_default():
            for name, value in scalars.items():
                tf.summary.scalar('scalars/' + name, value, step)
            for name, value in self._images.items():
                tf.summary.image(name, value, step)
            for name, value in self._videos.items():
                video_summary(name, value, step)
        self._writer.flush()

        # mlflow
        if self._log_mlflow:
            # The naming with _ prefix for main metrics is for more convenient mlflow browsing
            scalars['_step'] = step
            if 'loss_model' in scalars:
                scalars['_loss'] = scalars['loss_model']
            mlflow.log_metrics(scalars, step)

        self._scalars = {}
        self._scalars_mean = defaultdict(list)
        self._images = {}
        self._videos = {}

    def _compute_fps(self, step):
        if self._last_step is None:
            self._last_time = time.time()
            self._last_step = step
            return 0
        steps = step - self._last_step
        duration = time.time() - self._last_time
        self._last_time += duration
        self._last_step = step
        return steps / duration


def graph_summary(writer, step, fn, *args):
    def inner(*args):
        tf.summary.experimental.set_step(step.numpy().item())
        with writer.as_default():
            fn(*args)
    return tf.numpy_function(inner, args, [])


def video_summary(name, video, step=None, fps=20):
    name = name if isinstance(name, str) else name.decode('utf-8')
    if np.issubdtype(video.dtype, np.floating):
        video = np.clip(255 * video, 0, 255).astype(np.uint8)
    B, T, H, W, C = video.shape
    try:
        frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
        summary = tf1.Summary()
        image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
        image.encoded_image_string = encode_gif(frames, fps)
        summary.value.add(tag=name, image=image)
        tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
    except (IOError, OSError) as e:
        print('GIF summaries require ffmpeg in $PATH.', e)
        frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
        tf.summary.image(name, frames, step)


def encode_gif(frames, fps):
    from subprocess import Popen, PIPE
    h, w, c = frames[0].shape
    pxfmt = {1: 'gray', 3: 'rgb24'}[c]
    cmd = ' '.join([
        f'ffmpeg -y -f rawvideo -vcodec rawvideo',
        f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
        f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
        f'-r {fps:.02f} -f gif -'])
    proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in frames:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
    del proc
    return out