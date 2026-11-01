import yaml


def load_config(config_path='config.yaml'):
    """Loads a YAML config file and returns it as a dict."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
