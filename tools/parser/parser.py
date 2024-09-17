import argparse
import yaml
class ArgsParser():
    def __init__(self,):
        self.parser = argparse.ArgumentParser(description='script')
        self.parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    def parse_args(self,):
        """Return parsed args"""
        args = self.parser.parse_args()
        config = args.config
        with open(config, 'r') as file:
            config = yaml.load(file, Loader = yaml.FullLoader)
        return config