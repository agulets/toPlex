import os
from sysconfig import get_platform

from utils import get_platform

FFMPEG_MAC_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mac/')
FFMPEG_LINUX_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'linux/')
FFMPEG_WIN_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'win\\')


class ConfigException(Exception):
    def __init__(self, message):
        super().__init__(message)


def get_ffmpeg():
    if get_platform() == 'Darwin':
        return os.path.join(FFMPEG_MAC_PATH, 'ffmpeg')
    if get_platform() == 'Linux':
        return os.path.join(FFMPEG_LINUX_PATH, 'ffmpeg')
    if get_platform() == 'Windows':
        return os.path.join(FFMPEG_WIN_PATH, 'ffmpeg.exe')
    raise ConfigException(message=f"This os: '{get_platform()}' is not configured!")


def get_ffprobe():
    if get_platform() == 'Darwin':
        return os.path.join(FFMPEG_MAC_PATH, 'ffprobe')
    if get_platform() == 'Linux':
        return os.path.join(FFMPEG_LINUX_PATH, 'ffprobe')
    if get_platform() == 'Windows':
        return os.path.join(FFMPEG_WIN_PATH, 'ffprobe.exe')
    raise ConfigException(message=f"This os: '{get_platform()}' is not configured!")


class config:
    PLATFORM = get_platform()
    FFMPEG = get_ffmpeg()
    FFPROBE = get_ffprobe()


if __name__ == '__main__':
    print(get_ffmpeg())
    print(get_ffprobe())

