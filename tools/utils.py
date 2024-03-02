################################################################################

import json
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pycocotools.mask as mask_utils

################################################################################


"""Load COCO data from a JSON file."""


def load_coco_data(file_path):
    """
    Load COCO data from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The loaded COCO data.

    """
    with open(file_path) as f:
        return json.load(f)


################################################################################

"""Calculate various statistics such as counts, averages, and standard deviations."""


def calculate_stats(data):
    """
    Calculate various statistics from the given data.

    Args:
        data (dict): The input data containing images, annotations, and categories.

    Returns:
        tuple: A tuple containing the following statistics:
            - num_images (int): The number of images in the data.
            - num_annotations (int): The number of annotations in the data.
            - num_categories (int): The number of categories in the data.
            - category_counts (dict): A dictionary mapping category IDs to their respective counts.
            - category_sizes (dict): A nested dictionary mapping category IDs to their respective width and height lists.
            - object_sizes (dict): A nested dictionary mapping category IDs to their respective size categories and counts.
            - bbox_centers (dict): A nested dictionary mapping category IDs to their respective x and y coordinate lists.
    """
    num_images = len(data["images"])
    num_annotations = len(data["annotations"])
    num_categories = len(data["categories"])

    category_counts = defaultdict(int)
    category_sizes = defaultdict(lambda: defaultdict(list))
    object_sizes = defaultdict(lambda: defaultdict(int))
    bbox_centers = defaultdict(lambda: defaultdict(list))

    for annotation in data["annotations"]:
        category_id = annotation["category_id"]
        category_counts[category_id] += 1

        area = annotation["area"]
        size_category = (
            "small" if area < 32 * 32 else "medium" if area < 96 * 96 else "large"
        )
        object_sizes[category_id][size_category] += 1

        bbox = annotation["bbox"]
        width, height = bbox[2], bbox[3]
        category_sizes[category_id]["width"].append(width)
        category_sizes[category_id]["height"].append(height)

        center_x = bbox[0] + width / 2
        center_y = bbox[1] + height / 2
        bbox_centers[category_id]["x"].append(center_x)
        bbox_centers[category_id]["y"].append(center_y)

    return (
        num_images,
        num_annotations,
        num_categories,
        category_counts,
        category_sizes,
        object_sizes,
        bbox_centers,
    )


################################################################################


"""Write calculated statistics to a log file."""


def write_stats(log_file_path, data, stats):
    """
    Write statistics to a file.

    Args:
        log_file_path (str): The path to the log file.
        data (dict): The data containing categories information.
        stats (tuple): A tuple containing various statistics.

    Returns:
        None
    """

    (
        num_images,
        num_annotations,
        num_categories,
        category_counts,
        category_sizes,
        object_sizes,
        bbox_centers,
    ) = stats

    # Extract filename
    filename = os.path.basename(log_file_path)

    # Extract path
    log_dir = os.path.dirname(log_file_path)
    os.makedirs(log_dir, exist_ok=True)

    with open(log_file_path, "w") as log_file:
        log_file.write(f"Number of Images: {num_images}\n")
        log_file.write(f"Number of Annotations: {num_annotations}\n")
        log_file.write(f"Number of Categories: {num_categories}\n\n")

        log_file.write("Category Distribution:\n")
        for category in data["categories"]:
            category_id = category["id"]
            name = category["name"]
            count = category_counts[category_id]
            avg_width = np.mean(category_sizes[category_id]["width"])
            avg_height = np.mean(category_sizes[category_id]["height"])
            percentage = (count / num_annotations) * 100
            mean_center_x = np.mean(bbox_centers[category_id]["x"])
            mean_center_y = np.mean(bbox_centers[category_id]["y"])
            std_dev_center_x = np.std(bbox_centers[category_id]["x"])
            std_dev_center_y = np.std(bbox_centers[category_id]["y"])

            log_file.write(
                f"Category {name} (ID: {category_id}): Count = {count} ({percentage:.2f}%), "
                f"Avg Width = {avg_width:.2f}, Avg Height = {avg_height:.2f}, "
                f"Mean Center X = {mean_center_x:.2f}, Mean Center Y = {mean_center_y:.2f}, "
                f"Std Dev Center X = {std_dev_center_x:.2f}, Std Dev Center Y = {std_dev_center_y:.2f}\n"
            )

            log_file.write(f'  Small Objects: {object_sizes[category_id]["small"]}\n')
            log_file.write(f'  Medium Objects: {object_sizes[category_id]["medium"]}\n')
            log_file.write(f'  Large Objects: {object_sizes[category_id]["large"]}\n\n')


################################################################################

"""Find images without annotations."""


def find_images_without_annotations(data):
    """
    Find images without annotations in the given data.

    Args:
        data (dict): The data containing annotations and images.

    Returns:
        set: A set of image IDs that do not have any annotations.
    """
    annotated_image_ids = {annotation["image_id"] for annotation in data["annotations"]}
    all_image_ids = {image["id"] for image in data["images"]}
    images_without_annotations = all_image_ids - annotated_image_ids

    return images_without_annotations


################################################################################

"""Split COCO data into multiple parts and save each part."""


def split(
    data,
    num_parts,
    output_file,
    output_prefix="part",
):
    """
    Split the given data into multiple parts and save each part as a separate JSON file.

    Args:
        data (dict): The data to be split, containing "images", "annotations", and "categories".
        num_parts (int): The number of parts to split the data into.
        output_file (str): The path to the output file where the split data will be saved.
        output_prefix (str, optional): The prefix to be added to the output file names. Defaults to "part".

    Returns:
        None
    """
    total_images = len(data["images"])
    images_per_part = total_images // num_parts

    for i in range(num_parts):
        start_index = i * images_per_part
        end_index = (i + 1) * images_per_part if i < num_parts - 1 else total_images

        part_data = {
            "images": data["images"][start_index:end_index],
            "annotations": [
                ann
                for ann in data["annotations"]
                if ann["image_id"] in range(start_index + 1, end_index + 1)
            ],
            "categories": data["categories"],
        }

        # Extract filename
        filename = os.path.basename(output_file)
        filename = filename.split(".json")[0]
        # Extract path
        out_dir = os.path.dirname(output_file)
        os.makedirs(out_dir, exist_ok=True)

        part_filename = f"{out_dir}/{filename}{output_prefix}_{i + 1}.json"
        with open(part_filename, "w") as part_file:
            json.dump(part_data, part_file, indent=4)

        print(f"|__Part {i + 1} saved to: {part_filename}")


################################################################################

"""Remove specified categories and their annotations, then save the updated data."""


def remove_categories(data, category_ids, output_file):
    """
    Remove specified categories from the data and save the updated data to a file.

    Args:
        data (dict): The input data containing categories, annotations, and images.
        category_ids (list): The list of category IDs to be removed.
        output_file (str): The path to the output file where the updated data will be saved.

    Returns:
        None
    """
    # Filter out categories to be removed
    data["categories"] = [
        cat for cat in data["categories"] if cat["id"] not in category_ids
    ]

    # Filter out annotations of the specified categories
    data["annotations"] = [
        ann for ann in data["annotations"] if ann["category_id"] not in category_ids
    ]

    # Find images that are still referenced by the remaining annotations
    valid_image_ids = set(ann["image_id"] for ann in data["annotations"])

    # Filter out images that no longer have annotations
    data["images"] = [img for img in data["images"] if img["id"] in valid_image_ids]

    # Reassign new unique IDs to images and update in annotations
    image_id_mapping = {img["id"]: idx + 1 for idx, img in enumerate(data["images"])}
    for img in data["images"]:
        img["id"] = image_id_mapping[img["id"]]
    for ann in data["annotations"]:
        ann["image_id"] = image_id_mapping[ann["image_id"]]

    # Reassign new unique IDs to annotations
    for idx, ann in enumerate(data["annotations"]):
        ann["id"] = idx + 1

    # Extract filename
    filename = os.path.basename(output_file)
    # Extract path
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w") as updated_file:
        json.dump(data, updated_file, indent=4)

    print(f"|__Categorie(s) removed & saved to: {output_file}")


################################################################################

"""Draw bounding boxes or segmentation masks with category IDs on images."""


def draw_annotations(data, image_dir, output_dir):
    """
    Draw annotations on images based on the provided data.

    Args:
        data (dict): The data containing images, annotations, and categories.
        image_dir (str): The directory path where the images are located.
        output_dir (str): The directory path where the annotated images will be saved.
    """
    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    os.makedirs(output_dir, exist_ok=True)

    for image_info in images:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_path = f"{image_dir}/{file_name}"

        img = cv2.imread(image_path)
        h, w, c = img.shape
        # img_mask = np.zeros((h, w, 3), dtype = "uint8")

        for annotation in annotations:
            if annotation["image_id"] == image_id:
                category_id = annotation["category_id"]
                category_name = next(
                    cat["name"] for cat in categories if cat["id"] == category_id
                )

                bbox = annotation["bbox"]
                cv2.rectangle(
                    img,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    img,
                    category_name,
                    (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                # Alternatively, if segmentation information is available, you can draw masks
                if "segmentation" in annotation:
                    segmentation = annotation["segmentation"]

                    if len(segmentation) > 0:
                        if 'counts' in segmentation:
                            rle = segmentation
                            if isinstance(rle, dict):  # RLE format
                                mask = mask_utils.decode(rle)
                            else:  # Polygon format
                                # Create an empty mask and draw the polygons
                                mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                                
                                for seg in rle:
                                    poly = np.array(seg).reshape((-1, 1, 2))
                                    cv2.fillPoly(mask, [poly], 1)

                            if mask is not None:
                                # Convert mask to three channels and same data type as image
                                mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                                mask = mask * np.array([255, 255, 255], dtype=img.dtype)
                                # img = cv2.addWeighted(img, 1.0, mask, 0.5, 0)
                        
                        else:
                            for seg in segmentation:
                                # Create a white mask
                                mask = np.zeros_like(img, dtype=np.uint8)

                                # Draw the contour on the mask
                                cv2.drawContours(
                                    mask,
                                    [
                                        np.array([seg])
                                        .reshape((-1, 1, 2))
                                        .astype(np.int32)
                                    ],
                                    -1,
                                    (255, 255, 255),
                                    thickness=cv2.FILLED,
                                )
                            # Blend the mask with the original image using transparency
                        img = cv2.addWeighted(img, 1, mask, 0.5, 0)

        output_image_path = f"{output_dir}/{file_name}"
        cv2.imwrite(output_image_path, img)

        # print(f"Annotations drawn on {output_image_path}")


################################################################################

"""Plot a bar plot showing category-wise annotation counts."""


def plot_stats(data, plot_output_path):
    """
    Plot the category-wise annotation counts.

    Args:
        data (dict): The data containing annotations and categories.
        plot_output_path (str): The path to save the plot.

    Returns:
        None
    """
    # Extract filename
    filename = os.path.basename(plot_output_path)

    # Extract path
    plot_dir = os.path.dirname(plot_output_path)
    os.makedirs(plot_dir, exist_ok=True)

    category_counts = defaultdict(int)

    for annotation in data["annotations"]:
        category_id = annotation["category_id"]
        category_counts[category_id] += 1

    categories = data["categories"]
    category_names = [category["name"] for category in categories]
    category_ids = list(category_counts.keys())
    counts = [category_counts[cat_id] for cat_id in category_ids]

    # Choose a colormap, e.g., viridis
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(category_names)))
    plt.figure(figsize=(8, 6))
    bars = plt.bar(category_names, counts, color=colors, width=0.6)

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(count),
            ha="center",
        )

    # Remove spines (axes borders)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(False)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    # plt.xlabel('Category')
    # plt.ylabel('Count')
    # plt.title('Category-wise Annotation Counts')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plot_output_path)
    # plt.show()


################################################################################
"""Merge multiple COCO json(s) into a single json with rearranging the ids."""


def merge(file_paths, output_file):
    """
    Merge multiple COCO datasets into a single dataset.

    Args:
        file_paths (list): List of file paths to the COCO datasets.
        output_file (str): Path to the output file where the merged dataset will be saved.

    Returns:
        None
    """

    merged_data = {"images": [], "annotations": [], "categories": []}
    next_image_id = 1
    next_annotation_id = 1
    category_mapping = {}
    category_name_to_id = {}
    category_names = set()

    first_file = True
    for file_path in file_paths:
        data = load_coco_data(file_path)

        if first_file:
            for category in data["categories"]:
                category_mapping[category["id"]] = category["id"]
                category_names.add(category["name"])
                merged_data["categories"].append(category)
            first_file = False
        else:
            for category in data["categories"]:
                if category["name"] not in category_names:
                    category_mapping[category["id"]] = (
                        len(merged_data["categories"]) + 1
                    )
                    category["id"] = category_mapping[category["id"]]
                    category_names.add(category["name"])
                    merged_data["categories"].append(category)
                else:
                    category_mapping[category["id"]] = next(
                        filter(
                            lambda x: x["name"] == category["name"],
                            merged_data["categories"],
                        )
                    )["id"]

        # Update image IDs
        image_id_mapping = {
            image["id"]: next_image_id + i for i, image in enumerate(data["images"])
        }
        for image in data["images"]:
            image["id"] = image_id_mapping[image["id"]]
            merged_data["images"].append(image)

        # Update annotation IDs and map to new image and category IDs
        for annotation in data["annotations"]:
            annotation["id"] = next_annotation_id
            next_annotation_id += 1
            annotation["image_id"] = image_id_mapping[annotation["image_id"]]
            annotation["category_id"] = category_mapping[annotation["category_id"]]
            merged_data["annotations"].append(annotation)

        next_image_id += len(data["images"])

    # Remove duplicate categories
    merged_data["categories"] = [
        dict(t) for t in {tuple(d.items()) for d in merged_data["categories"]}
    ]
    # Sort categories by ID in ascending order
    merged_data["categories"] = sorted(merged_data["categories"], key=lambda x: x["id"])

    # Extract filename
    filename = os.path.basename(output_file)
    # Extract path
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"|__Merged json(s) written to: {output_file}")


################################################################################

"""Filter annotations by min max height/width"""


def filter_annotation_img_size(
    data, min_width, min_height, max_width, max_height, output_file
):
    """
    Filter annotations based on image dimensions and save the filtered annotations to a JSON file.

    Args:
        data (dict): The annotation data containing images and annotations.
        min_width (int): The minimum width threshold for filtering annotations.
        min_height (int): The minimum height threshold for filtering annotations.
        max_width (int): The maximum width threshold for filtering annotations.
        max_height (int): The maximum height threshold for filtering annotations.
        output_file (str): The path to the output JSON file.

    Returns:
        None
    """
    # Mapping from image IDs to their respective widths and heights
    image_dimensions = {
        image["id"]: (image["width"], image["height"]) for image in data["images"]
    }

    # Filter out annotations with image sizes outside the thresholds
    data["annotations"] = [
        ann
        for ann in data["annotations"]
        if min_width <= image_dimensions[ann["image_id"]][0] <= max_width
        and min_height <= image_dimensions[ann["image_id"]][1] <= max_height
    ]

    # Find images that are still referenced by the remaining annotations
    valid_image_ids = set(ann["image_id"] for ann in data["annotations"])

    # Filter out images that no longer have annotations
    data["images"] = [img for img in data["images"] if img["id"] in valid_image_ids]

    # Reassign new unique IDs to images and update in annotations
    image_id_mapping = {img["id"]: idx + 1 for idx, img in enumerate(data["images"])}
    for img in data["images"]:
        img["id"] = image_id_mapping[img["id"]]
    for ann in data["annotations"]:
        ann["image_id"] = image_id_mapping[ann["image_id"]]

    # Reassign new unique IDs to annotations
    for idx, ann in enumerate(data["annotations"]):
        ann["id"] = idx + 1

    # Extract filename
    filename = os.path.basename(output_file)
    # Extract path
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Filtered annotation json written to: {output_file}")


################################################################################

"""Filter annotations by bbox min/max height/width"""


def filter_annotation_bbox_size(
    data, min_width, min_height, max_width, max_height, output_file
):
    """
    Filter out annotations based on bounding box size constraints.

    Args:
        data (dict): The input data containing annotations and images.
        min_width (int): The minimum width of the bounding box.
        min_height (int): The minimum height of the bounding box.
        max_width (int): The maximum width of the bounding box.
        max_height (int): The maximum height of the bounding box.
        output_file (str): The path to the output file where the filtered annotation JSON will be written.

    Returns:
        None
    """

    def is_bbox_within_thresholds(annotation):
        _, _, bbox_width, bbox_height = annotation["bbox"]
        return (min_width < bbox_width < max_width) and (
            min_height < bbox_height < max_height
        )

    data["annotations"] = [
        ann for ann in data["annotations"] if is_bbox_within_thresholds(ann)
    ]

    # Find images that are still referenced by the remaining annotations
    valid_image_ids = set(ann["image_id"] for ann in data["annotations"])

    # Filter out images that no longer have annotations
    data["images"] = [img for img in data["images"] if img["id"] in valid_image_ids]

    # Reassign new unique IDs to images and update in annotations
    image_id_mapping = {img["id"]: idx + 1 for idx, img in enumerate(data["images"])}
    for img in data["images"]:
        img["id"] = image_id_mapping[img["id"]]
    for ann in data["annotations"]:
        ann["image_id"] = image_id_mapping[ann["image_id"]]

    # Reassign new unique IDs to annotations
    for idx, ann in enumerate(data["annotations"]):
        ann["id"] = idx + 1

    # Extract filename
    filename = os.path.basename(output_file)
    # Extract path
    out_dir = os.path.dirname(output_file)
    os.makedirs(out_dir, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Filtered annotation json written to: {output_file}")
