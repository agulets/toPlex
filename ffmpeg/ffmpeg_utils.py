import os
import subprocess
import traceback
from collections import namedtuple
from ffmpeg.ffmpeg_config import config


AudioInfo = namedtuple('AudioInfo', [
    'file_name',
    'streams_count',
    'format_name',
    'duration',
    'size',
    'stream0_codec_name',
    'stream0_type',
    'stream0_bits_per_sample',
    'stream0_bit_rate',
    'stream0_sample_format',
    'stream0_channels_count',
    'stream0_sample_rate',
    'error'
])


def exec_ffprobe(*params):
    error = None
    stdout = None
    ffprobe = config.FFPROBE

    if not os.path.exists(ffprobe):
        error = f"No ffprobe found: {ffprobe}"

    if not os.access(ffprobe, os.X_OK):
        error = f"FFprobe found but it's not have executable permissions!"

    if not error:
        try:
            exec_result = subprocess.Popen([ffprobe] + list(params), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = exec_result.communicate()
            error = stderr
        except Exception as e:
            error = f"Ffprobe error:\n{''.join(traceback.format_tb(e.__traceback__))}"

    return stdout, error


def exec_ffmpeg(*params):
    error = None
    stdout = None
    ffmpeg = config.FFMPEG

    if not os.path.exists(ffmpeg):
        error = f"No ffmpeg found: {ffmpeg}"

    if not os.access(ffmpeg, os.X_OK):
        error = f"FFprobe found but it's not have executable permissions!"

    if not error:
        try:
            exec_result = subprocess.Popen([ffmpeg] + list(params), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stdout, stderr = exec_result.communicate()
            error = stderr
        except Exception as e:
            error = f"Ffmpeg error:\n{''.join(traceback.format_tb(e.__traceback__))}"

    # TODO Need check stdout correctly!
    if not error and stdout.decode().find('Permission denied'):
        error = stdout.decode()

    return stdout, error


# def get_audio_info(audio_file):
#     file_name = audio_file
#     streams_count = None
#     format_name = None
#     duration = None
#     size = None
#     stream0_codec_name = None
#     stream0_codec_type = None
#     stream0_bits_per_sample = None
#     stream0_bit_rate = None
#     stream0_sample_format = None
#     stream0_channels_count = None
#     stream0_sample_rate = None
#     error = None
#
#     stdout = None
#
#     if not os.path.isfile(file_name):
#         error = f"FileNotFoundError: {file_name}"
#
#     if not error:
#         stdout, stderr = exec_ffprobe('-v', 'error', '-show_streams', '-show_format', '-of', 'json', file_name)
#         if stderr:
#             error = f"ffprobe returned next error: \n{stderr}\n"
#
#         if not error and 'Invalid data found when processing input' in stdout.decode():
#             error = f"ffprobe returned next error: Invalid data found when processing input: '{file_name}'"
#
#     if not error:
#         try:
#             out_data = ast.literal_eval(stdout.decode())
#             streams_count = out_data['format']['nb_streams']
#             format_name = out_data['format']['format_name']
#             duration = out_data['format']['duration']
#             size = out_data['format']['size']
#             stream0_codec_name = out_data['streams'][0]['codec_name']
#             stream0_codec_type = out_data['streams'][0]['codec_type']
#             stream0_bits_per_sample = out_data['streams'][0]['bits_per_sample']
#             stream0_bit_rate = out_data['streams'][0]['bit_rate']
#             stream0_sample_format = out_data['streams'][0]['sample_fmt']
#             stream0_sample_rate = out_data['streams'][0]['sample_rate']
#             stream0_channels_count = int(out_data['streams'][0]['channels'])
#         except Exception as e:
#             error = f"Problem in result interpreting:\n{''.join(traceback.format_tb(e.__traceback__))}"
#
#     return AudioInfo(file_name, streams_count, format_name, duration, size, stream0_codec_name, stream0_codec_type,
#                      stream0_bits_per_sample, stream0_bit_rate, stream0_sample_format, stream0_channels_count,
#                      stream0_sample_rate, error)


if __name__ == "__main__":
    f_res, f_err = exec_ffmpeg('-help')
    print(f_res)
    print(f"ERROR:\n\n{f_err}")
