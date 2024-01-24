################################################################################
# Example : Perform annotation check/modification/draw for coco format json
# Copyright (c) 2024 - Neelanjan Bhowmik
# License: https://github.com/NeelBhowmik/AnnotateMate/blob/main/LICENSE
################################################################################

import argparse
from tabulate import tabulate
import tools.utils as utils

################################################################################
# Main function


def main():
    """
    Entry point of the COCO annotation helper script.
    Parses command line arguments, performs various operations on COCO annotation files,
    and displays the results.

    Usage:
        python main.py [annotation_file ...] [--log_file LOG_FILE] [--split_parts SPLIT_PARTS]
                       [--remove_categories REMOVE_CATEGORIES ...] [--output_annotation OUTPUT_ANNOTATION]
                       [--draw] [--merge] [--image_dir IMAGE_DIR] [--output_draw OUTPUT_DRAW]
                       [--stat] [--plot_stat] [--output_plot OUTPUT_PLOT]

    Arguments:
        annotation_file         Path to the COCO annotation file(s)
        
    Optional Arguments:
        --log_file              Path to the output log file (default: logs/coco_stats_log.log)
        --split                 Number of parts to split the JSON file into (default: 1)
        --remove_categories     List of category IDs to remove
        --output_annotation     Directory to save output annotation file (default: logs/out.json)
        --draw                  Draw bounding boxes or masks
        --merge                 Merge multiple coco-json files
        --image_dir             Directory to image directory
        --output_draw           Directory to save bbox/mask draw images (default: output_draw)
        --stat                  Analyse annotation stats
        --plot_stat             Plot category-wise annotation counts
        --output_plot           File path to save stats plot image (default: logs/coco_plot.png)
    """
    parser = argparse.ArgumentParser(description="COCO annotation helper.")
    parser.add_argument(
        "annotation_file",
        nargs="+",
        help="Path to the COCO annotation file(s)",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="logs/coco_stats_log.log",
        help="Path to the output log file",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="Number of parts to split the JSON file into",
    )
    parser.add_argument(
        "--remove_categories",
        nargs="+",
        type=int,
        help="List of category IDs to remove",
    )
    parser.add_argument(
        "--output_annotation",
        type=str,
        default="logs/out.json",
        help="Directory to save output annotation file",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Draw bounding boxes or masks",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge multiple coco-json files",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Directory to image directory",
    )
    parser.add_argument(
        "--output_draw",
        type=str,
        default="output_draw",
        help="Directory to save bbox/mask draw images",
    )
    parser.add_argument(
        "--stat",
        action="store_true",
        help="Analyse annotation stats",
    )
    parser.add_argument(
        "--plot_stat",
        action="store_true",
        help="Plot category-wise annotation counts",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="logs/coco_plot.png",
        help="File path to save stats plot image",
    )
    args = parser.parse_args()
    t_val = []
    for arg in vars(args):
        t_val.append([arg, getattr(args, arg)])
    print(tabulate(t_val, ["input", "value"], tablefmt="psql"))

    print(f"|__Input coco-json file(s): {args.annotation_file}")

    # Merge multiple coco-json files
    if args.merge:
        if len(args.annotation_file) > 1:
            print(f"|__Merging {len(args.annotation_file)} files")
            utils.merge(args.annotation_file, args.output_annotation)
        else:
            print(f"|__Only one annotation file provided!")
            exit(0)

    else:
        # Read the coco json file
        coco_data = utils.load_coco_data(args.annotation_file[0])

        # Calculate statistics
        if args.stat:
            print(f"|__Generating stats: {args.log_file}")
            stats = utils.calculate_stats(coco_data)

            # Write statistics to a log file
            utils.write_stats(args.log_file, coco_data, stats)
            print(f"|__Annotation stats: {args.log_file}")

        # Plot category-wise annotation counts
        if args.plot_stat:
            print(f"|__Plotting category stats: {args.output_plot}")
            utils.plot_stats(coco_data, args.output_plot)

        # Split the json file into multiple parts
        if args.split > 1:
            print(f"|__Splitting into {args.split_parts} parts")
            utils.split(
                coco_data, args.split_parts, args.output_annotation
            )

        # Remove specified categories and save the updated data
        if args.remove_categories:
            print(f"|__Removing categories: {args.remove_categories}")
            utils.remove_categories(
                coco_data, args.remove_categories, args.output_annotation
            )

        # Draw bounding boxes or masks with category IDs
        if args.draw:
            print(f"|__Drawing bbox/segm annotation: {args.output_draw}")
            utils.draw_annotations(coco_data, args.image_dir, args.output_draw)

    print("\n\n[Done]")


################################################################################

if __name__ == "__main__":
    main()
