import os
import re
import ssl
import sys
import time
import shutil
import socket
import zipfile
import logging
import inspect
import hashlib
import datetime
import platform
import traceback
import contextlib
from tqdm import tqdm
from pathlib import Path
from contextlib import closing
from urllib.parse import urlparse
from collections import namedtuple
from multiprocessing import get_context
from logging.handlers import RotatingFileHandler
from multiprocessing.queues import Queue as BaseQueue
from chardet.universaldetector import UniversalDetector


TRUTH = ['true', '1', 'truth', 'yes']

MODIFICATION = 'modification'
CREATION = 'creation'
AGE_CONDITIONS = [MODIFICATION, CREATION]

OLDER = 'older'
YOUNGER = 'younger'
OLDER_OR_YOUNGER_CONDITIONS = [OLDER, YOUNGER]

FILE_PARAMS = namedtuple('FILE_PARAMS', [
        'full_file_path',
        'file_dir',
        'file_name',
        'file_extension',
        'file_coding'
])


def to_bool(item):
    return str(item).lower() in TRUTH


def get_var_name(var):
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class TimeItException(Exception):
    def __init__(self, message, logger=None):
        self.message = message
        logger = logger if logger else logging.getLogger('root')
        logger.error('TimeItException: %s' % self.message)


def attemperator(exceptions=Exception, max_attempts=-1, delay=0.01, logger=None):
    logger = logger if logger else logging.getLogger('root')

    def _decorator(func):
        def _wrapper(*args, **kwargs):
            if max_attempts > 0:
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as exc:
                        if attempt <= max_attempts - 1:
                            logger.warning(f"Have exception with next type {type(exc)}. "
                                           f"It was {attempt} attempt in {max_attempts}. "
                                           f"Wait {delay}s for new attempt.")
                        else:
                            raise
                    time.sleep(delay)
            else:
                while True:
                    try:
                        return func(*args, **kwargs)
                    except exceptions as exc:
                        logger.warning(f"Have exception with next type {type(exc)}. "
                                       f"Wait '{delay}'s for new attempt.")
                    time.sleep(delay)
        return _wrapper
    return _decorator


# TODO Need force GC initiate with deleting objects if fail - may memory leak if use with async or in threads
# TODO Need "infinity" replace to -1, or add ENUM for it
def trying(max_tries,
           sleep_time,
           raise_last_exception=False,
           logger=None,
           ):
    logger = logger if logger else logging.getLogger('root')

    def decorator(func):
        def wrapper(*function_args, **function_kwargs):
            if max_tries == 'infinity':
                while True:
                    try:
                        return func(*function_args, **function_kwargs)
                    except Exception:
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        trace_error = f"\t\t".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                        error = f"Error while trying to execute: {func.__name__}\n" \
                                f"\tError type: {exc_type}\n" \
                                f"\tError value: {exc_value}\n" \
                                f"\tError trace:\n" \
                                f"{'-' * 40}StartTrace{'-' * 40}\n" \
                                f"{trace_error}\n" \
                                f"{'-' * 40}EndTrace{'-' * 40}\n" \
                                f"Next try of {func.__name__} execution will be after {sleep_time} sec!"
                        logger.error(error)
                        time.sleep(sleep_time)
            else:
                for try_number in range(1, max_tries + 1):
                    try:
                        return func(*function_args, **function_kwargs)
                    except Exception as e:
                        exc_type, exc_value, exc_tb = sys.exc_info()
                        trace_error = f"".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                        error = f"Error while trying to execute: {func.__name__}\n" \
                                f"\tError type: {exc_type}\n" \
                                f"\tError value: {exc_value}\n" \
                                f"\tError trace: \n" \
                                f"{'-' * 40}StartTrace{'-' * 40}\n" \
                                f"{trace_error}\n" \
                                f"{'-' * 40}EndTrace{'-' * 40}\n"
                        if max_tries - try_number > 0:
                            logger.error(f"{error}\n"
                                         f"{max_tries - try_number} tries remaining! Sleep for {sleep_time} sec.")
                            time.sleep(sleep_time)
                        else:
                            logger.error(error)
                            if raise_last_exception:
                                raise e
            return None
        return wrapper
    return decorator


def get_host_port_from_url(url):
    if len(url.split('://')) == 2:
        url_parse = urlparse(url)
        if not url_parse.hostname or not url_parse.port:
            raise ValueError(f"Cant get host or port from url:'{url}'"
                             f" Host:'{url_parse.hostname}'"
                             f" Port:'{url_parse.port}'")
        return url_parse.hostname, url_parse.port
    if len(url.split('://')) == 1:
        url_parts = url.split(':')
        if len(url_parts) != 2:
            raise ValueError(f"Cant get host or port from url:'{url}' Cant split it by ':' symbol correctly.")
        if not url_parts[1].isdigit():
            raise ValueError(f"Cant get host or port from url:'{url}' Cant get port from url part {url_parts[1]}")
        return url_parts[0], int(url_parts[1])

    raise ValueError(f"Cant get host or port from url:'{url}' Cant split it by '://' correctly.")


def socket_check(host, port, timeout=2):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(timeout)
        try:
            connection_result = sock.connect_ex((host, port))
        except socket.gaierror as connection_result:
            sock.close()
            return False, f"Connection problem: {connection_result}."
        if connection_result == 0:
            sock.close()
            return True, f"Connection OK!"
        else:
            sock.close()
            return False, f"Connection problem: {connection_result}!"


def wait_for_socket(host, port, timeout=2, sleep_time=10, logger=None):
    logger = logger if logger else logging.getLogger('root')
    is_socket_alive = False
    logger.debug(f"Connection check for '{host}:{port}'")
    while not is_socket_alive:
        if not host or not port:
            logger.error(f"Connection check for '{host}:{port}' failed! Check host and port parameter.")
            time.sleep(sleep_time)
        else:
            is_socket_alive, connection_description = socket_check(host=host, port=port, timeout=timeout)
            if not is_socket_alive:
                logger.error(f"Connection check for '{host}:{port}' failed! Description:'{connection_description}'")
                logger.info(f"Wait for socket '{host}:{port}', sleep for {sleep_time} sec.")
                time.sleep(sleep_time)
    logger.debug(f"Connection check for '{host}:{port}' - OK!")


class NamedQueue(BaseQueue):
    def __init__(self, name):
        self.name = name
        super().__init__(ctx=get_context())

    def get_name(self):
        return self.name


def get_queue_by_task(task_queues, task_config, logger):
    while True:
        for queue in task_queues:
            logger.info(f"Try to get queue with name {task_config.name}")
            if task_config.name == queue.get_name():
                return queue
        logger.warn(f"Queue with {task_config.name} not initialized yet! "
                    f"Sleep for {task_config.queue_filling_sleep_time} sec")
        time.sleep(task_config.queue_filling_sleep_time)


def get_today(date_time_format=None):
    if date_time_format:
        return datetime.datetime.strftime(datetime.datetime.today(), date_time_format)
    return datetime.datetime.today()


def get_logger_by_params_and_make_log_folder(log_name, log_dir, log_file_name, log_size, log_file_count, log_level,
                                             formatter=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{os.path.join(log_dir, log_file_name)}.log"
    formatter = formatter if formatter else '%(asctime)s\t%(levelname)s\t%(processName)' \
                                            's:%(funcName)s:%(lineno)d:\t%(message)s'
    log_formatter = logging.Formatter(formatter)
    log_file_handler = RotatingFileHandler(filename=log_filename, maxBytes=log_size, backupCount=log_file_count)
    log_console_handler = logging.StreamHandler(sys.stdout)
    log_file_handler.setFormatter(log_formatter)
    log_console_handler.setFormatter(log_formatter)
    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)
    logger.addHandler(log_file_handler)
    logger.addHandler(log_console_handler)

    return logger


def get_datetime_by_string(date_time_str, date_time_format):
    return datetime.datetime.strptime(date_time_str, date_time_format)


def get_delta_days(date1, date2=None):
    date2 = get_today() if not date2 else date2
    delta = date2 - date1
    return delta.days


def get_delta_minutes(date1, date2=None):
    date2 = get_today() if not date2 else date2
    delta = date2 - date1
    return (delta.days * 24 * 60) + delta.seconds // 60


def get_platform():
    return platform.system()


def get_node_id():
    return hashlib.sha1(os.getcwd().encode() + platform.node().encode()).hexdigest()


def get_node(conf):
    try:
        if getattr(conf, 'CLIENT_NODE_ID'):
            return conf.CLIENT_NODE_ID
        else:
            return f"{platform.node()}_{get_node_id()}"
    except AttributeError:
        return f"{platform.node()}_{get_node_id()}"


def get_modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def get_creation_date(filename):
    t = os.path.getctime(filename)
    return datetime.datetime.fromtimestamp(t)


@contextlib.contextmanager
def time_it(msg_format, assertion_time_in_s=None, start_msg_format=None, logger=None):
    logger = logger if logger else logging.getLogger('root')
    if start_msg_format:
        logger.debug(start_msg_format.format(time.time()))
    start = time.time()
    yield
    if assertion_time_in_s:
        assertion_time_in_ms = assertion_time_in_s * 1000
        diff_time = time.time() * 1000 - (start * 1000)
        message = f'Time assertion error. Time limit is: {assertion_time_in_s}s. {msg_format.format(diff_time / 1000)}'
        if diff_time > assertion_time_in_ms:
            raise TimeItException(message)
    execution_time = time.time() - start
    msg = msg_format.format(round(execution_time, 3))
    logger.info(msg)


def zip_file(input_file_path, logger=None, output_directory=None, delete_original=False):
    logger = logger if logger else logging.getLogger('root')
    with time_it('Zip_File: File zipping finished at: {} sec', logger=logger):
        input_file_name = os.path.basename(input_file_path)
        input_directory = os.path.dirname(input_file_path)
        output_directory = output_directory or input_directory
        output_file_path = os.path.join(output_directory, input_file_name) + '.zip'
        logger.debug(f'Output file path: {output_file_path}')
        logger.debug(f'Zipper_file started for file: {input_file_path}')
        assert os.path.exists(input_file_path), "Zip_File: File not exists"
        with zipfile.ZipFile(output_file_path, 'w') as zip_data_file:
            try:
                zip_data_file.write(filename=input_file_path,
                                    arcname=get_file_name(input_file_path),
                                    compress_type=zipfile.ZIP_DEFLATED)
                zip_data_file.close()
                if delete_original:
                    os.remove(input_file_path)
            except Exception as e:
                logger.exception(f'Exception:\n\t{e}')
                raise
        return output_file_path


def try_zip_file(input_file_path, logger=None, output_directory=None, delete_original=False,
                 max_tries=10, sleep_time=10, raise_last_exception=True):
    logger = logger if logger else logging.getLogger('root')

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def try_zip_it(_input_file_path=input_file_path,
                   _logger=logger,
                   _output_directory=output_directory,
                   _delete_original=delete_original):
        zip_file(input_file_path=_input_file_path, logger=_logger,
                 output_directory=_output_directory, delete_original=_delete_original)

    try_zip_it(_input_file_path=input_file_path, _logger=logger,
               _output_directory=output_directory, _delete_original=delete_original)


def get_file_coding(file):
    detector = UniversalDetector()
    with open(file, 'rb') as fh:
        for line in fh:
            detector.feed(line)
            if detector.done:
                break
        detector.close()
    return detector.result['encoding']


def get_full_file_path(file):
    return str(Path(file).resolve())


def get_file_dir(file):
    return os.path.sep.join(str(get_full_file_path(file)).split(os.path.sep)[:-1]) + os.path.sep


def get_file_name(file):
    return str(get_full_file_path(file)).split(os.path.sep)[-1]


def get_file_extension(file):
    _, file_ext = os.path.splitext(str(get_full_file_path(file)))
    return file_ext


def is_folder_empty(folder):
    return [f for f in os.listdir(folder) if not f.startswith('.')] == []


def get_full_file_names_by_pattern_in_directory(directory, pattern='', include_sub_folders=True, logger=None):
    logger = logger if logger else logging.getLogger('root')
    logger.debug(f"Start search files in directory:'{directory}' by pattern:'{pattern}' "
                 f"{f'include' if include_sub_folders else 'exclude'} subfolders.")
    all_files = []
    files_by_pattern = []
    folder = os.path.abspath(directory)
    if include_sub_folders:
        for root, dirs, files in os.walk(folder, topdown=False):
            for name in files:
                all_files.append(os.path.join(root, name))
    else:
        include = set(directory)
        for root, dirs, files in os.walk(folder, topdown=True):
            dirs[:] = [d for d in dirs if d in include]
            for name in files:
                all_files.append(os.path.join(root, name))

    if pattern:
        for file in all_files:
            try:
                if re.fullmatch(string=get_file_name(file), pattern=pattern):
                    files_by_pattern.append(file)
            except Exception as e:
                logger.error(f"Problem in regexp finder:\n\t String:{file}\n\t Pattern:{pattern}\n\t Exception: {e} ")
        return files_by_pattern

    logger.debug(f"Found {len(all_files)} files.")
    return all_files


def get_sub_dirs(path, names_only=False):
    if names_only:
        if [x[0] for x in os.walk(path)]:
            return [x for x in os.walk(path)][0][1]
        else:
            return []
    return [x[0] for x in os.walk(path)][1:]


def remove_folder(folder_path):
    shutil.rmtree(folder_path)


def get_file_params(file, with_file_coding=False):
    if with_file_coding:
        return FILE_PARAMS(
            full_file_path=get_full_file_path(file=file),
            file_dir=get_file_dir(file=file),
            file_name=get_file_name(file=file),
            file_extension=get_file_extension(file=file),
            file_coding=get_file_coding(file=file),
            )
    else:
        return FILE_PARAMS(
            full_file_path=get_full_file_path(file=file),
            file_dir=get_file_dir(file=file),
            file_name=get_file_name(file=file),
            file_extension=get_file_extension(file=file),
            file_coding=None
            )


def is_all_elements_matches_in_list(lst):
    for element in lst:
        if element != lst[0]:
            return False
    return True


class ListLoopIterator:
    def __iter__(self):
        return self

    def __init__(self, lst):
        if not isinstance(lst, list):
            raise TypeError
        if len(lst) == 0:
            raise StopIteration
        self.lst = lst
        self.len = len(lst)
        self.count = -1

    def __next__(self):
        self.count += 1
        return self.lst[self.count % self.len]


def _sort_by_alphabet(input_str):
    return input_str[0]


def _sort_by_length(input_str):
    return len(input_str)


def get_all_sub_folders(directory):
    return [x[0] for x in os.walk(directory)]


def delete_all_empty_folders_in_directory(directory, delete_source_if_empty=False, logger=None):
    logger = logger if logger else logging.getLogger('root')
    logger.info(f"Start deleting empty folders from '{directory}' and all subdirectories.")

    @trying(max_tries=10,
            sleep_time=10,
            raise_last_exception=True,
            logger=logger)
    def try_delete_folder(_dir):
        logger.debug(f"Try delete folder '{_dir}'")
        shutil.rmtree(_dir)
        logger.debug(f"Empty folder '{_dir}' was deleted!")

    all_subdirectories = get_all_sub_folders(directory)

    if not delete_source_if_empty:
        all_subdirectories.remove(directory)

    all_subdirectories = list(set(all_subdirectories))
    all_subdirectories.sort(key=_sort_by_length, reverse=True)

    logger.info(f"Found {len(all_subdirectories)} subdirectories.")
    for folder in all_subdirectories:
        if is_folder_empty(folder):
            try_delete_folder(folder)
    logger.info("Deleting empty subdirectories was completed.")


def move_file(source_file, destination, max_tries=3, sleep_time=10, raise_last_exception=True, logger=None):
    logger = logger if logger else logging.getLogger('root')

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def _try_move_file(src, dst):
        logger.debug(f"Move file: {src} to {dst}")
        shutil.move(src=src, dst=dst)

    _try_move_file(src=source_file, dst=destination)


def copy_file(source_file, destination, max_tries=3, sleep_time=10, raise_last_exception=True, logger=None):
    logger = logger if logger else logging.getLogger('root')

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def try_copy_file(src, dst):
        logger.debug(f"Copy file: {src} to {dst}")
        shutil.copy(src=src, dst=dst)

    try_copy_file(src=source_file, dst=destination)


def delete_file(_file, max_tries=3, sleep_time=10, raise_last_exception=True, logger=None):
    logger = logger if logger else logging.getLogger('root')

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def try_delete_file(src):
        logger.debug(f"Delete file: '{src}'")
        os.remove(src)
        logger.debug(f"File: '{src}' Deleted OK!")

    try_delete_file(_file)


def trying_write_data_to_file(data, file, logger=None, max_tries=3, sleep_time=5, raise_last_exception=True,
                              if_exists="rewrite"):
    logger = logger if logger else logging.getLogger('root')

    rewrite_options = ['rewrite', 'skip', 'raise_exception']

    if if_exists not in rewrite_options:
        raise ValueError(f"Unknown 'if_exists' option: '{if_exists}'! It can be only '{rewrite_options}'.")

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def _write_data_to_file(_data, _file, _if_exists, _logger=None):
        _logger = logger if logger else logging.getLogger('root')

        if os.path.isfile(_file):
            if _if_exists == "rewrite":
                logger.warning(f"File '{_file}' already exists and will be delete and rewritten!")
                delete_file(_file=file, max_tries=1, raise_last_exception=True, logger=_logger)

            if _if_exists == "skip":
                logger.warning(f"File '{_file}' already exists and will skipped!")
                return

            if _if_exists == "raise_exception":
                raise OSError(f"File '{_file}' already exist!")

        with open(_file, 'wb') as out_file:
            out_file.write(_data)
        logger.debug(f"File '{file}' was written ok.")

    logger.debug(f"Start write file '{file}'.")
    _write_data_to_file(_data=data, _file=file, _if_exists=if_exists, _logger=logger)


def move_files_to_separate_folders(full_files_paths,
                                   output_file_dirs,
                                   logger=None,
                                   max_tries=3,
                                   sleep_time=5,
                                   raise_last_exception=True,
                                   delete_source_files=True,
                                   delete_source_folders=False,
                                   delete_empty_source_folders=False):
    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def __try_move_file(src, dst):
        logger.debug(f"Move file: '{src}' to '{dst}'")
        shutil.move(src=src, dst=dst)

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def __try_copy_file(src, dst):
        logger.debug(f"Copy file: '{src}' to '{dst}'")
        shutil.copy(src=src, dst=dst)

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def __try_delete_folder(_folder):
        logging.debug(f"Delete folder '{_folder}'")
        shutil.rmtree(_folder)

    logger = logger if logger else logging.getLogger('root')
    logger.info(f'Start moving files from many sources to {output_file_dirs}')
    output_directory_iterator = ListLoopIterator(output_file_dirs)
    print(f"Moving files progress:")
    time.sleep(1)

    all_sources_dirs = []

    for file_path in tqdm(full_files_paths):
        output_directory = next(output_directory_iterator)
        destination_file_path = os.path.join(output_directory, get_file_name(file_path))
        source_file_dir = get_file_dir(file_path)
        all_sources_dirs.append(source_file_dir)

        if delete_source_files:
            __try_move_file(src=f'{file_path}', dst=f'{destination_file_path}')
        else:
            __try_copy_file(src=f'{file_path}', dst=f'{destination_file_path}')

    all_sources_dirs = list(set(all_sources_dirs))
    all_sources_dirs.sort(key=_sort_by_length, reverse=True)

    if delete_source_folders:
        for folder in all_sources_dirs:
            __try_delete_folder(_folder=folder)

    else:
        if delete_empty_source_folders:
            for folder in all_sources_dirs:
                if is_folder_empty(folder):
                    __try_delete_folder(_folder=folder)


def try_move_file(src, dst, max_tries=10, sleep_time=10, raise_last_exception=True, logger=None):
    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def _try_move_file(_src, _dst, _logger=None):
        _logger = _logger if _logger else logging.getLogger('root')
        _logger.debug(f"Move file: '{_src}' to '{_dst}'")
        shutil.move(src=_src, dst=_dst)

    _try_move_file(_src=src, _dst=dst, _logger=logger)


def try_add_postfix_to_filename(file, postfix, max_tries=3, sleep_time=5, raise_last_exception=True, logger=None):
    logger = logger if logger else logging.getLogger('root')

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def add_postfix_to_filename(_file, _postfix, _logger):
        if os.path.isfile(_file):
            file_name = get_file_name(file=_file)
            file_path = get_file_dir(file=_file)
            new_file_name = f"{file_name}{_postfix}"
            new_file = os.path.join(file_path, new_file_name)
            os.rename(_file, new_file)
            _logger.debug(f"File '{_file}' renamed to '{new_file}'")
            return new_file
        else:
            raise OSError(f"File '{_file}' doesn't exist!")

    return add_postfix_to_filename(_file=file, _postfix=postfix, _logger=logger)


def try_del_postfix_from_filename(file, postfix, max_tries=3, sleep_time=5,
                                  delete_if_exists=True, raise_last_exception=True, logger=None):
    logger = logger if logger else logging.getLogger('root')

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def del_postfix_from_filename(_file, _postfix, _logger):
        if os.path.isfile(_file):
            file_name = get_file_name(file=_file)
            file_path = get_file_dir(file=_file)
            new_file_name = file_name.replace(_postfix, "")
            if file_name == new_file_name:
                raise ValueError(f"File '{_file}' doesn't contain postfix '{_postfix}'")
            new_file = os.path.join(file_path, new_file_name)
            if os.path.isfile(new_file):
                if delete_if_exists:
                    _logger.warning(f"File '{new_file}' already exists and will be rewritten!")
                    delete_file(_file=new_file,
                                max_tries=1,
                                logger=_logger)
                else:
                    raise OSError(f"File '{new_file}' already exists!")
            os.rename(_file, new_file)
            _logger.debug(f"File '{_file}' renamed to '{new_file}'")
            return new_file
        else:
            raise OSError(f"File '{_file}' doesn't exist!")

    return del_postfix_from_filename(_file=file, _postfix=postfix, _logger=logger)


def copy_file_to_many_folders(input_full_file_path, output_file_dirs, logger=None,
                              delete_source_file=False, delete_source_folders=False, raise_last_exception=True,
                              delete_source_folders_if_empty=False, max_tries=5, sleep_time=10):
    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def try_copy_file(src, dst):
        logger.debug(f"Copy file: '{src}' to '{dst}'")
        shutil.copy(src=src, dst=dst)

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def try_delete_folder(_folder):
        logging.debug(f"Delete folder '{_folder}'")
        shutil.rmtree(_folder)

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def try_delete_file(file):
        logging.debug(f"Delete file '{file}'")
        os.remove(file)

    logger = logger if logger else logging.getLogger('root')

    for output_directory in output_file_dirs:
        try_copy_file(src=f'{input_full_file_path}', dst=f'{output_directory}')

    if delete_source_file:
        try_delete_file(input_full_file_path)

    folder = get_file_dir(input_full_file_path)
    if delete_source_folders:
        try_delete_folder(_folder=folder)
        logging.info(f"Folder '{folder}' deleted!")
    else:
        if delete_source_folders_if_empty and is_folder_empty(folder):
            try_delete_folder(_folder=folder)
            logging.info(f"Folder '{folder}' deleted because it was empty!")


def get_files_by_pattern_and_age_condition(path, logger=None, pattern='', exclude_patterns=None,
                                           condition_of_age=CREATION, age_in_minutes=0,
                                           older_or_younger_condition=OLDER, include_sub_folders=True):
    logger = logger if logger else logging.getLogger('root')

    if condition_of_age not in AGE_CONDITIONS:
        logger.error(f"Unknown condition_of_age parameter: {condition_of_age}!")
        return

    if older_or_younger_condition not in OLDER_OR_YOUNGER_CONDITIONS:
        logger.error(f"Unknown older_or_younger_condition parameter: {older_or_younger_condition}!")
        return

    if not os.path.exists(path=path):
        logger.error(f"Path '{path}' doesn't exist or not accessible!")

    all_files_by_pattern = get_full_file_names_by_pattern_in_directory(directory=path,
                                                                       pattern=pattern,
                                                                       logger=logger,
                                                                       include_sub_folders=include_sub_folders)

    all_files_for_exclude = []
    if exclude_patterns:
        for exclude_pattern in exclude_patterns:
            files_for_exclude_by_pattern = []
            for file_ in all_files_by_pattern:
                if re.fullmatch(string=get_file_name(file_), pattern=exclude_pattern):
                    files_for_exclude_by_pattern.append(file_)
            all_files_for_exclude.extend(files_for_exclude_by_pattern)

    files_by_all_patterns = []
    for check_file in all_files_by_pattern:
        if check_file not in all_files_for_exclude:
            files_by_all_patterns.append(check_file)

    files_by_older_condition = []
    if not age_in_minutes:
        return files_by_all_patterns
    else:
        for file in files_by_all_patterns:
            try:
                file_age = 0
                if condition_of_age == MODIFICATION:
                    file_age = get_modification_date(file)
                if condition_of_age == CREATION:
                    file_age = get_creation_date(file)

                logger.debug(f"File: '{file}' have age: {get_delta_minutes(file_age)} ")
                delta_in_minutes = get_delta_minutes(file_age)
                if delta_in_minutes >= age_in_minutes:
                    files_by_older_condition.append(file)
            except (IOError, OSError) as e:
                logger.warning(f"IO problem: '{e}'")

        if older_or_younger_condition == OLDER:
            return files_by_older_condition
        if older_or_younger_condition == YOUNGER:
            return [_file for _file in files_by_all_patterns if _file not in files_by_older_condition]


def delete_old_files_in_folder(age_in_minutes, folder, logger, pattern='', condition_of_age=CREATION,
                               max_tries=5, sleep_time=10, raise_last_exception=True, include_sub_folders=True):
    logger = logger if logger else logging.getLogger('root')

    @trying(max_tries=max_tries,
            sleep_time=sleep_time,
            raise_last_exception=raise_last_exception,
            logger=logger)
    def try_delete_file(file_path):
        logger.debug(f"Trying  delete file: '{file_path}'")
        os.remove(file_path)

    all_files_by_pattern = get_full_file_names_by_pattern_in_directory(directory=folder,
                                                                       pattern=pattern,
                                                                       logger=logger,
                                                                       include_sub_folders=include_sub_folders)

    for file in all_files_by_pattern:
        try:
            if condition_of_age in AGE_CONDITIONS:
                file_creation_date = get_creation_date(file)
                file_last_modification_date = get_modification_date(file)

                file_age = 0
                if condition_of_age == MODIFICATION:
                    file_age = file_last_modification_date
                if condition_of_age == CREATION:
                    file_age = file_creation_date

                logger.debug(f"File: '{file}' have age: {get_delta_minutes(file_age)} ")
                delta_in_minutes = get_delta_minutes(file_age)
                if delta_in_minutes >= age_in_minutes:
                    try_delete_file(file)
                    logger.info(f"File: '{file}' was DELETED!  Creation date {file_creation_date}  "
                                f"Last modification date {file_last_modification_date}  "
                                f"Delete reason: age till {condition_of_age} is {delta_in_minutes}")
            else:
                logger.error(f"Unknown condition_of_age parameter: {condition_of_age}!")
                return
        except (IOError, OSError) as e:
            logger.warning(f"IO problem: {e}")
    return


def get_ssl_ctx(verify_cert_path=None, client_cert_path=None, client_cert_key_path=None, logger=None):
    logger = logger if logger else logging.getLogger('root')
    try:
        ssl_ctx = ssl.create_default_context()
        if not verify_cert_path:

            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
        else:
            ssl_ctx.check_hostname = True
            ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            ssl_ctx.load_verify_locations(cafile=verify_cert_path)
            ssl_ctx.load_cert_chain(client_cert_path, client_cert_key_path)
        return ssl_ctx
    except Exception as ssl_ctx_e:
        logger.exception(f"Problem in getting SSL: {ssl_ctx_e}")
        raise ssl_ctx_e


if __name__ == '__main__':
    # import random
    #
    # log = logging.getLogger('root')
    # log.setLevel(10)
    # log.addHandler(logging.StreamHandler(sys.stdout))
    # # wait_for_socket(host='tplss-terra0008.sigma.sbrf.ru', port=9090, logger=log)
    #
    # @attemperator(exceptions=ValueError,
    #               max_attempts=3,
    #               delay=1,
    #               logger=log)
    # def _attemp_test():
    #     print('!')
    #     # if random.randint(1, 6) == 1:
    #     #     raise OSError('Test_os_error')
    #     raise ValueError('Test_error')
    #
    # _attemp_test()

    url = 'http://127.0.0.1:1111'
    print(get_host_port_from_url(url))
