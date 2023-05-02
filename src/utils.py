import json
import pickle
import pandas as pd


class Utils:

    def __init__(self, data=None):
        self.data = data
        self.read_functions = {
            'json': lambda path: json.load(open(path)),
            'csv': lambda path: pd.read_csv(path),
            'pkl': lambda path: pickle.load(open(path, 'rb')),
            'xlsx': lambda path: pd.read_excel(path),
            'txt': lambda path: open(path, 'r', encoding='utf-8').read()
        }

        self.write_functions = {
            'json': lambda path, data: json.dump(data, open(path, 'w')),
            'csv': lambda path, data: pd.DataFrame(data).to_csv(path),
            'pkl': lambda path, data: pickle.dump(data, open(path, 'wb')),
            'xlsx': lambda path, data: pd.DataFrame(data).to_excel(path),
            'txt': lambda path, data: open(path, 'w', encoding='utf-8').write(data)
        }

    def save_to_file(self, file_path: str, file_type: str=None):
        file_type = file_type or file_path.split('.')[-1]
        self.write_functions[file_type](file_path, self.data)
        return f'Saved to {file_path} as type {file_type}'

    def read_from_file(self, file_path: str, file_type: str=None):
        file_type = file_type or file_path.split('.')[-1]
        self.data = self.read_functions[file_type](file_path)
        return self.data
