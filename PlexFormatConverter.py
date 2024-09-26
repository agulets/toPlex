import os.path

from tqdm import tqdm
from ffmpeg.ffmpeg_utils import exec_ffmpeg, exec_ffprobe
from utils import get_full_file_names_by_pattern_in_directory, get_file_name, delete_file

if __name__ == '__main__':

    INPUT_FOLDER = 'TMP/sample_video_in/'
    OUTPUT_FOLDER = 'TMP/out/'

    # f_res, f_err = exec_ffmpeg('-help')
    # print(f_res)
    # print(f"ERROR:\n\n{f_err}")
    all_video_files = get_full_file_names_by_pattern_in_directory(INPUT_FOLDER, pattern=r'.*.(avi|wmv|mp4|m4v|mov|mkv)')
    # for file in all_video_files:
    #     print(file)

    # f_res, f_err = exec_ffmpeg('-i', f"{all_video_files[0]}")
    # print(f_res.decode())
    # print(f"ERROR:{f_err}")


    # for file_for_convert in tqdm(all_video_files):
    for file_for_convert in all_video_files:
        f_res, f_err = '', ''
        print(f"Input file {file_for_convert}")

        output_file = os.path.join(os.path.abspath(OUTPUT_FOLDER), get_file_name(file_for_convert))
        print(f"Output file {output_file}")

        # delete output file if already exist
        if os.path.exists(output_file):
            print(f"File '{output_file}' already exist and will be removed!")
            delete_file(output_file, sleep_time=1)


        f_res, f_err = exec_ffmpeg('-i', f"{file_for_convert}", '-c:v', 'libx264', '-c:a', 'aac', '-crf', '15', f'{output_file}')
        # print(f_res.decode())
        if f_err:
            print(f" ---- ERROR: -----\n{f_err}\n --------------------------------")

