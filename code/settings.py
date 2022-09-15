import ast
import configparser
from collections.abc import Mapping


class Settings(Mapping):
    def __init__(self, setting_file='settings.ini'):
        config = configparser.ConfigParser()
        config.read(setting_file)
        self.settings_dict = _parse_values(config)

    def __getitem__(self, key):
        return self.settings_dict[key]

    def __len__(self):
        return len(self.settings_dict)

    def __iter__(self):
        return self.settings_dict.items()


def _parse_values(config):
    config_parsed = {}
    for section in config.sections():
        config_parsed[section] = {}
        for key, value in config[section].items():
            config_parsed[section][key] = ast.literal_eval(value)
    return config_parsed
