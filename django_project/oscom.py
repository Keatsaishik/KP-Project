# List of os commands
import os
import pprint
from datetime import datetime
import shutil

def remove_directory_tree(directory):
    try:
        shutil.rmtree(directory)
        print(f"Directory '{directory}' and its contents removed successfully.")
    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
    except Exception as e:
        print(f"Error occurred while removing the directory: {str(e)}")

def get_file_modification_time(file_name):
    try:
        mod_time = os.stat(file_name).st_mtime
        return datetime.fromtimestamp(mod_time)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found.")
    except Exception as e:
        print(f"Error occurred while getting file modification time: {str(e)}")

def get_all_commands():
    try:
        pprint.pprint(dir(os))
    except Exception as e:
        print(f"Error occurred while getting all commands: {str(e)}")

def get_all_help():
    try:
        pprint.pprint(help(os))
    except Exception as e:
        print(f"Error occurred while getting all help: {str(e)}")

def get_current_directory():
    try:
        return os.getcwd()
    except Exception as e:
        print(f"Error occurred while getting the current directory: {str(e)}")

def change_current_directory(directory):
    try:
        os.chdir(directory)
    except Exception as e:
        print(f"Error occurred while changing the current directory: {str(e)}")

def make_directories(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(f"Error occurred while making directories: {str(e)}")

def remove_directory(directory):
    try:
        os.rmdir(directory)
    except Exception as e:
        print(f"Error occurred while removing the directory: {str(e)}")


def get_environment_variable(variable_name):
    try:
        return os.environ.get(variable_name)
    except Exception as e:
        print(f"Error occurred while getting the environment variable: {str(e)}")

def join_paths(path1, path2):
    try:
        return os.path.join(path1, path2)
    except Exception as e:
        print(f"Error occurred while joining paths: {str(e)}")

def check_path_exists(path):
    try:
        return os.path.exists(path)
    except Exception as e:
        print(f"Error occurred while checking if the path exists: {str(e)}")

def is_directory(path):
    try:
        return os.path.isdir(path)
    except Exception as e:
        print(f"Error occurred while checking if the path is a directory: {str(e)}")

def is_file(path):
    try:
        return os.path.isfile(path)
    except Exception as e:
        print(f"Error occurred while checking if the path is a file: {str(e)}")
# Call the functions as needed in the main function
