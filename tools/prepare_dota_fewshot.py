#!/usr/bin/env python3
"""Split DOTA text annotations into disjoint base and few-shot image sets.

The source image directory is shared by both splits; this script only writes
annotation folders and a split manifest, so a large image corpus is not
duplicated.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Create DOTA annotations_base and annotations_few_shot folders '
            'by selecting a fixed number of images per class.'))
    parser.add_argument('--images', required=True, type=Path,
                        help='Folder containing source DOTA images.')
    parser.add_argument('--annotations', required=True, type=Path,
                        help='Folder containing per-image DOTA .txt files.')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output directory for the two annotation folders.')
    parser.add_argument(
        '--shots-per-class', '--few-shot-per-class', type=int, default=18,
        help='Number of distinct images sampled for each selected class (default: 18).')
    parser.add_argument(
        '--sample-classes', nargs='+', default=None,
        help='Classes used for balanced sampling. Defaults to all observed classes.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for repeatable image selection.')
    return parser.parse_args()


def parse_annotation(path):
    """Return original lines and the classes present in a DOTA label file."""
    text = path.read_text(encoding='utf-8')
    labels = set()
    lines = text.splitlines(keepends=True)
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 9:
            raise ValueError(
                f'{path}:{line_number}: expected 8 polygon coordinates and a class name.')
        labels.add(fields[8])
        if len(fields) == 9:
            content = line.rstrip('\r\n')
            line_ending = line[len(content):]
            lines[line_number - 1] = content + ' 0' + line_ending
    return ''.join(lines), labels


def validate_inputs(images_dir, annotations_dir):
    if not images_dir.is_dir():
        raise FileNotFoundError(f'Image directory does not exist: {images_dir}')
    if not annotations_dir.is_dir():
        raise FileNotFoundError(
            f'Annotation directory does not exist: {annotations_dir}')

    image_stems = {
        path.stem for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    ann_files = sorted(annotations_dir.glob('*.txt'))
    if not ann_files:
        raise ValueError(f'No DOTA .txt annotations found in {annotations_dir}')

    missing_images = [path.stem for path in ann_files if path.stem not in image_stems]
    if missing_images:
        preview = ', '.join(missing_images[:10])
        raise ValueError(
            f'{len(missing_images)} annotation files have no matching image in '
            f'{images_dir}; examples: {preview}')
    return ann_files


def write_split(out_dir, names, annotation_data):
    out_dir.mkdir(parents=True, exist_ok=False)
    for stem in sorted(names):
        _, source_path, _ = annotation_data[stem]
        (out_dir / source_path.name).write_text(
            annotation_data[stem][0], encoding='utf-8')


def main():
    args = parse_args()
    if args.shots_per_class <= 0:
        raise ValueError('--shots-per-class must be a positive integer.')

    images_dir = args.images.expanduser().resolve()
    annotations_dir = args.annotations.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    ann_files = validate_inputs(images_dir, annotations_dir)

    annotation_data = {}
    observed_classes = set()
    for path in ann_files:
        text, labels = parse_annotation(path)
        annotation_data[path.stem] = (text, path, labels)
        observed_classes.update(labels)

    sample_classes = args.sample_classes or sorted(observed_classes)
    unknown_classes = sorted(set(sample_classes) - observed_classes)
    if unknown_classes:
        raise ValueError(
            'Requested sampling classes do not occur in the source annotations: '
            + ', '.join(unknown_classes))

    candidates_by_class = defaultdict(list)
    for stem, (_, _, labels) in annotation_data.items():
        for class_name in labels:
            candidates_by_class[class_name].append(stem)

    rng = random.Random(args.seed)
    few_shot_stems = set()
    sampled_image_counts = {}
    for class_name in sample_classes:
        candidates = sorted(candidates_by_class[class_name])
        rng.shuffle(candidates)
        selected = candidates[:args.shots_per_class]
        few_shot_stems.update(selected)
        sampled_image_counts[class_name] = len(selected)

    all_stems = set(annotation_data)
    base_stems = all_stems - few_shot_stems
    base_image_count = sum(
        bool(annotation_data[stem][2]) for stem in base_stems)
    if base_image_count == 0:
        raise ValueError(
            'The selected support images leave no non-empty base annotations; '
            'reduce --shots-per-class or provide more source images.')

    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = output_dir / 'annotations_base'
    few_shot_dir = output_dir / 'annotations_few_shot'
    manifest_path = output_dir / 'split_manifest.json'
    if base_dir.exists() or few_shot_dir.exists() or manifest_path.exists():
        raise FileExistsError(
            f'Output split files already exist under {output_dir}. '
            'Choose a new --output directory so previous labels are preserved.')
    write_split(base_dir, base_stems, annotation_data)
    write_split(few_shot_dir, few_shot_stems, annotation_data)

    support_class_images = Counter()
    support_class_objects = Counter()
    for stem in few_shot_stems:
        _, _, labels = annotation_data[stem]
        for class_name in labels:
            support_class_images[class_name] += 1
        text = annotation_data[stem][0]
        for line in text.splitlines():
            fields = line.split()
            if len(fields) >= 9:
                support_class_objects[fields[8]] += 1

    manifest = {
        'images_dir': str(images_dir),
        'source_annotations_dir': str(annotations_dir),
        'shots_per_class': args.shots_per_class,
        'sample_classes': list(sample_classes),
        'seed': args.seed,
        'num_base_images': base_image_count,
        'num_few_shot_images': sum(
            bool(annotation_data[stem][2]) for stem in few_shot_stems),
        'sampled_images_per_class': sampled_image_counts,
        'few_shot_images_per_class': dict(sorted(support_class_images.items())),
        'few_shot_objects_per_class': dict(sorted(support_class_objects.items())),
        'base_annotations': str(base_dir),
        'few_shot_annotations': str(few_shot_dir),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')

    print(
        f'Base annotations: {base_dir} '
        f'({manifest["num_base_images"]} non-empty images)')
    print(
        f'Few-shot annotations: {few_shot_dir} '
        f'({manifest["num_few_shot_images"]} images)')
    print(f'Shared images: {images_dir}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
