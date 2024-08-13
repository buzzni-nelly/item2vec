
import glob
import itertools
import json
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import clients.ceph
import directories
from item2vec import vocab

SOURCE_DIR = directories.data.as_posix()

DESTINATION_DIR = "/item2vec/20240701-20240812"

ceph = clients.ceph.CephClient()
ceph.upload_dir(directories.data.as_posix(), "/item2vec/20240701-20240812")
