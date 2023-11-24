import os

import omegaconf


def env_with_prefix(key: str, prefix: str, default: str) -> str:
    value = os.getenv(key)
    if value:
        return prefix + value
    return default


def register_custom_resolvers():
    omegaconf.OmegaConf.register_new_resolver("ADD", lambda x, y: x + y)
    omegaconf.OmegaConf.register_new_resolver("SUB", lambda x, y: x - y)
    omegaconf.OmegaConf.register_new_resolver("MUL", lambda x, y: x * y)
    omegaconf.OmegaConf.register_new_resolver("DIV", lambda x, y: x / y)
    omegaconf.OmegaConf.register_new_resolver("INTDIV", lambda x, y: x // y)
    omegaconf.OmegaConf.register_new_resolver("env_with_prefix", env_with_prefix)
