# -*- coding: utf-8 -*-
#
# Copyright (c) 2020, MultiplEYE co. ltd.
# All rights reserved.
#
# utils.config.py
#
# Author: Changhee Won (changhee.won@multipleye.co)
#
#
from utils.common import *
import yaml
from string import Template

def load_yaml_config_file(path: str) -> Dict[str, Any]:
    config = yaml.safe_load(open(path))
    return config

def update_flags_recursively(name: str, val: Any, root: Dict[str, Any]):
    def expand_str_variable(var: str, node: Dict[str, Any]) -> str:
        if var.count('$') <= 0: return var
        return Template(var).substitute(node)
    if type(val) == dict:
        for k, v in val.items():
            update_flags_recursively(name + '_' + k, v, root)
    else:
        if not name in FLAGS: return
        flag = FLAGS[name]
        if flag.using_default_value:
            if type(val) == str:
                val = expand_str_variable(val, root)
            flag.value = val
            VLOG(1, '- yaml %s -> %s', name, val)
        else:
            VLOG(1, '- FLAGS_%s <- %s', name, flag.value)

def load_yaml_config_and_sync_flags_recursively(
        config: Dict[str, Any], parent_name: str):
    keys_to_load: List[str] = []
    for key, val in config.items():
        if key.startswith('load_'):
            keys_to_load.append(key)
        else:
            update_flags_recursively(parent_name + key, val, config)
    return config

def load_app_yaml_config_and_sync_flags(app_path: str, parent: str = '') -> bool:
    yaml_file = osp.splitext(app_path)[0] + '.yaml'
    if parent: yaml_file = osp.join(parent, yaml_file)
    if not osp.exists(yaml_file):
        LOG_WARNING('Config file does not exist: ' + yaml_file)
        return False
    config = load_yaml_config_file(yaml_file)
    load_yaml_config_and_sync_flags_recursively(config, "")
    return True

def load_app_yaml_config_from_custom_path_and_sync_flags(yaml_file: str) -> bool:
    if not osp.exists(yaml_file):
        LOG_WARNING('Config file does not exist: ' + yaml_file)
        return False
    config = load_yaml_config_file(yaml_file)
    load_yaml_config_and_sync_flags_recursively(config, "")
    return True
