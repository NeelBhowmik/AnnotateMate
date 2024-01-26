
<div align="center">

**AnnotateMate: Open-source tool to analyse, modify COCO format annotation**

---
</div>

<!-- A open-source tool to support annotation files. -->
### :peacock: Features
<small>

- [x] **One script does it all**
    - [x] [analyse](#detective-analyse): Analyse and generate stats & plot
    - [x] [draw](#art-draw): Draw bbox/segm from annotation file 
    - [x] [split](#scissors-split): Split annotation file
    - [x] [merge](#beers-merge): Merge multiple annotations files
    - [x] [remove categories](#broom-remove-categories): Remove categories/corresponding annotations from annotation file
    - [x] [filter annotation](#mag-filter-annotation ): Filter annotations by bbox min/max height/width
- [x] Supported annotation format: **[COCO](https://cocodataset.org/#format-data)**

:space_invader: This repo is constantly updated with the new features - check in regularly for updates!

</small>


## :wrench: Installation
Install the requirements
~~~
pip3 install -r requirements.txt
~~~

## :plate_with_cutlery: Getting started

Run the `main.py` with different command line options:
~~~
main.py [-h] [--log_file LOG_FILE] [--split SPLIT] [--remove_categories REMOVE_CATEGORIES [REMOVE_CATEGORIES ...]]
               [--output_annotation OUTPUT_ANNOTATION] [--draw] [--merge] [--image_dir IMAGE_DIR]
               [--output_draw OUTPUT_DRAW] [--stat] [--plot_stat] [--output_plot OUTPUT_PLOT]
               annotation_file [annotation_file ...]

COCO annotation helper.

positional arguments:
  annotation_file       Path to the COCO annotation file(s)

options:
  -h, --help            show this help message and exit
  --log_file LOG_FILE   Path to the output log file
  --split SPLIT         Number of parts to split the JSON file into
  --remove_categories REMOVE_CATEGORIES [REMOVE_CATEGORIES ...]
                        List of category IDs to remove
  --output_annotation OUTPUT_ANNOTATION
                        Directory to save output annotation file
  --draw                Draw bounding boxes or masks
  --merge               Merge multiple coco-json files
  --image_dir IMAGE_DIR
                        Directory to image directory
  --output_draw OUTPUT_DRAW
                        Directory to save bbox/mask draw images
  --stat                Analyse annotation stats
  --plot_stat           Plot category-wise annotation counts
  --output_plot OUTPUT_PLOT
                        File path to save stats plot image
~~~

### :detective: Analyse 
Analyse and generate statisctics and plot from annotation file 
~~~
python3 main.py \
    <path/to/coco-json> \
    --stat \
    --log_file <path/to/save/stats-log.log> \
    --plot_stat \
    --output_plot <path/to/save/plot-file.png>
~~~

### :art: Draw 
Draw bbox/segm from annotation file  
~~~
python3 main.py \
    <path/to/coco-json> \
    --draw \
    --image_dir <path/to/image/dir> \
    --output_draw <path/to/save/draw/bbox-segm/dir>
~~~

### :scissors: Split
Split annotation file into *n*-parts 
~~~
python3 main.py \
    <path/to/coco-json> \
    --split 4 \
    --output_annotation <path/to/save/split/annotation-josn>
~~~

### :beers: Merge 
Merge multiple coco annotations files
~~~
python3 main.py \
    <path/to/coco-json-2> <path/to/coco-json-2> \
    --merge \
    --output_annotation <path/to/save/merge/annotation-josn>
~~~

### :broom: Remove Categories 
Remove categories/corresponding annotations from annotation file
~~~
python3 main.py \
    <path/to/coco-json> \
    --remove_categories 2 3 \
    --output_annotation <path/to/save/remove-cat/annotation-josn>
~~~

### :mag: Filter Annotation 
Filter annotations by bbox min/max height/width
~~~
python3 main.py \
    <path/to/coco-json> \
    coco_data,
    --min_width 20
    --min_height 20
    --max_width 150
    --max_height 160
    --output_annotation <path/to/save/filter/annotation-josn>
~~~

## :frog: Reference
If you use this repo and like it, use this to cite it:
```tex
@misc{annotatemate,
      title={AnnotateMate: Open-source tool to analyse, modify COCO format annotation},
      author={Neelanjan Bhowmik},
      year={2024},
      url={https://github.com/NeelBhowmik/AnnotateMate}
    }
```