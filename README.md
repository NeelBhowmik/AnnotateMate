# Annotation help

A open-source tool to support annotation files.

## Features
- coco format json check

## Getting started

### Supported file
- [&check;] json
- [~] yml

### Supported annotation format
- [&check;] coco
- [~] kitti

1. To check/validate coco format json annotation:
    ~~~
    coco_check.py [-h] [--jsonfile JSONFILE] [--logfile LOGFILE]

    optional arguments:
    -h, --help           show this help message and exit
    --jsonfile JSONFILE  Input json file path
    --logfile LOGFILE    Log file path
    ~~~

## TODO
- [] visualise
- [] merge
- [] convert
- [] extract category
- [] split