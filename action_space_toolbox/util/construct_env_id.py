import re


def construct_env_id(base_env_name: str, control_mode: str) -> str:
    version_str = re.findall("-v[0-9]+", base_env_name)[-1]
    return f"{base_env_name[:-len(version_str)]}_{control_mode}{version_str}"
