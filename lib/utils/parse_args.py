#!/usr/bin/python3
#coding:utf-8

"""
    Author: Xie Song
    Email: 18406508513@163.com
    
    Copyright: Xie Song
    License: MIT
"""

import yaml
from easydict import EasyDict


def parse_args(config):
    """

    :param parser:
    :return: args
    """
    global args

    args = config
    if args.config_file:
        with open(args.config_file) as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
            data = EasyDict(config['common'])
            delattr(args, 'config_file')
            arg_dict = args.__dict__
            # if key exists in args, replace the existed key value, else add key value into args
            for key, value in data.items():
                # nested group
                if isinstance(value, dict):
                    for k, v in value.items():
                        if k in arg_dict.keys():
                            arg_dict[k] = v
                arg_dict[key] = value
    return args