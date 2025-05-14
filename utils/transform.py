import os
import hashlib

def rename_files_to_md5(directory):
    for subdir, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(subdir, file)
            md5_hash = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    md5_hash.update(chunk)
            md5_filename = f"{md5_hash.hexdigest()}.{file.split('.')[-1]}"
            new_file_path = os.path.join(subdir, md5_filename)
            os.rename(file_path, new_file_path)
            print(f'Renamed: {file_path} to {new_file_path}')

rename_files_to_md5('./datasets')  # Change this to your target directory path

