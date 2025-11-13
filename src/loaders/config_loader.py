import yaml
from pathlib import Path
from types import MappingProxyType

def deep_update(base, overrides):
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base

def resolve_paths(config, base_dir):
    base_dir = Path(base_dir)
    for section in config.values():
        if isinstance(section, dict):
            for key, value in section.items():
                if isinstance(value, str) and ("/" in value or "\\" in value):
                    section[key] = str((base_dir / value).resolve())
    return config

def freeze_config(config):
    if isinstance(config, dict):
        return MappingProxyType({k: freeze_config(v) for k, v in config.items()})
    elif isinstance(config, list):
        return tuple(freeze_config(v) for v in config)
    return config

def load_config(config_path, overrides=None, base_dir="."):
    # Load main config
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply override layers
    if overrides:
        for override_file in overrides:
            override_path = Path(override_file)
            with open(override_path, "r") as f:
                override = yaml.safe_load(f)
            config = deep_update(config, override)

    # Resolve relative paths
    config = resolve_paths(config, base_dir=base_dir)

    # Freeze to avoid accidental modification
    config = freeze_config(config)

    return config
