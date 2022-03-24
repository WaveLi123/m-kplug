#!/usr/bin/env python 
# -*- coding: utf-8 -*-

import os
import shutil


def file_exist(infile):
    return os.path.isfile(infile)


def dir_exist(indir):
    return os.path.isdir(indir)


def create_dir(indir):
    if not dir_exist(indir):
        os.makedirs(indir)


def delete_file(file_path):
    try:
        if file_exist(file_path):
            os.remove(file_path)
    except Exception as error:
        print(file_path, error)


def delete_dir(file_dir):
    shutil.rmtree(file_dir)


def check_dir(dir_path):
    if not dir_exist(dir_path):
        create_dir(dir_path)


def rename_file(in_file, new_in_file):
    try:
        os.rename(in_file, new_in_file)
    except Exception as error:
        print(in_file, error)
