#!/usr/bin/env python3
"""Check whether a downloaded Validation split is complete.

Examples:
  python3 check_validation_split.py 3dod --download_dir data
  python3 check_validation_split.py upsampling --download_dir data
  python3 check_validation_split.py raw --download_dir data --raw_dataset_assets mov annotation mesh
"""

import argparse
import csv
import os
from collections import defaultdict

HIGRES_DEPTH_ASSET_NAME = 'highres_depth'
MISSING_3DOD_ASSETS_VIDEO_IDS = {
    '47334522', '47334523', '42897421', '45261582', '47333152', '47333155',
    '48458535', '48018733', '47429677', '48458541', '42897848', '47895482',
    '47333960', '47430089', '42899148', '42897612', '42899153', '42446164',
    '48018149', '47332198', '47334515', '45663223', '45663226', '45663227'
}
DEFAULT_RAW_DATASET_ASSETS = [
    'mov', 'annotation', 'mesh', 'confidence', 'highres_depth', 'lowres_depth',
    'lowres_wide.traj', 'lowres_wide', 'lowres_wide_intrinsics', 'ultrawide',
    'ultrawide_intrinsics', 'vga_wide', 'vga_wide_intrinsics'
]


def read_split_csv(path, split='Validation'):
    ids = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('fold') == split:
                ids.append(str(row['video_id']))
    return sorted(set(ids))


def read_raw_metadata(path):
    rows = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[str(row['video_id'])] = row
    return rows


def expected_raw_files(video_id, assets, raw_metadata):
    names = []
    for asset in assets:
        if asset == HIGRES_DEPTH_ASSET_NAME:
            md = raw_metadata.get(str(float(video_id))) or raw_metadata.get(video_id)
            is_in_upsampling = False
            if md is not None:
                is_in_upsampling = str(md.get('is_in_upsampling', '')).strip().lower() == 'true'
            if not is_in_upsampling:
                continue

        if asset in {
            'confidence', 'highres_depth', 'lowres_depth', 'lowres_wide', 'lowres_wide_intrinsics',
            'ultrawide', 'ultrawide_intrinsics', 'vga_wide', 'vga_wide_intrinsics'
        }:
            names.append(asset + '.zip')
        elif asset == 'mov':
            names.append(f'{video_id}.mov')
        elif asset == 'mesh':
            if video_id not in MISSING_3DOD_ASSETS_VIDEO_IDS:
                names.append(f'{video_id}_3dod_mesh.ply')
        elif asset == 'annotation':
            if video_id not in MISSING_3DOD_ASSETS_VIDEO_IDS:
                names.append(f'{video_id}_3dod_annotation.json')
        elif asset == 'lowres_wide.traj':
            if video_id not in MISSING_3DOD_ASSETS_VIDEO_IDS:
                names.append('lowres_wide.traj')
        else:
            raise ValueError(f'Unsupported raw asset: {asset}')
    return names


def zip_asset_complete(dst_dir, zip_name):
    zip_path = os.path.join(dst_dir, zip_name)
    marker_path = os.path.join(dst_dir, f'.{zip_name}.unzip_complete')
    extracted_dir = zip_path[:-4] if zip_name.endswith('.zip') else None

    if os.path.isfile(marker_path):
        return True
    # Backward compatibility for old downloads before marker support.
    if os.path.isdir(extracted_dir):
        return True
    if os.path.isfile(zip_path):
        return True
    return False


def check_dataset(dataset, download_dir, raw_assets):
    if dataset == '3dod':
        split_csv = 'threedod/3dod_train_val_splits.csv'
        ids = read_split_csv(split_csv)
        base = os.path.join(download_dir, '3dod', 'Validation')
        missing = []
        for vid in ids:
            zip_name = f'{vid}.zip'
            if not zip_asset_complete(base, zip_name):
                missing.append((vid, zip_name, base))
        return ids, missing

    if dataset == 'upsampling':
        split_csv = 'depth_upsampling/upsampling_train_val_splits.csv'
        ids = read_split_csv(split_csv)
        base = os.path.join(download_dir, 'upsampling', 'Validation')
        missing = []
        for vid in ids:
            zip_name = f'{vid}.zip'
            if not zip_asset_complete(base, zip_name):
                missing.append((vid, zip_name, base))
        # val_attributes.csv is a validation-only extra file.
        if not os.path.isfile(os.path.join(base, 'val_attributes.csv')):
            missing.append(('Validation', 'val_attributes.csv', base))
        return ids, missing

    if dataset == 'raw':
        split_csv = 'raw/raw_train_val_splits.csv'
        metadata_csv = os.path.join(download_dir, 'raw', 'metadata.csv')
        ids = read_split_csv(split_csv)
        if not os.path.isfile(metadata_csv):
            raise FileNotFoundError(
                f'Raw completeness check requires metadata at {metadata_csv}. '
                'Please run download_data.py raw first (or place metadata there).'
            )
        raw_metadata = read_raw_metadata(metadata_csv)

        missing = []
        for vid in ids:
            base = os.path.join(download_dir, 'raw', 'Validation', vid)
            expected_files = expected_raw_files(vid, raw_assets, raw_metadata)
            for name in expected_files:
                path = os.path.join(base, name)
                if name.endswith('.zip'):
                    if not zip_asset_complete(base, name):
                        missing.append((vid, name, base))
                elif not os.path.isfile(path):
                    missing.append((vid, name, base))
        return ids, missing

    raise ValueError(f'Unsupported dataset: {dataset}')


def main():
    parser = argparse.ArgumentParser(description='Check whether Validation split download is complete.')
    parser.add_argument('dataset', choices=['3dod', 'upsampling', 'raw'])
    parser.add_argument('--download_dir', default='data')
    parser.add_argument('--raw_dataset_assets', nargs='+', choices=DEFAULT_RAW_DATASET_ASSETS)

    args = parser.parse_args()

    raw_assets = args.raw_dataset_assets
    if args.dataset == 'raw' and not raw_assets:
        parser.error('--raw_dataset_assets is required when dataset=raw')

    ids, missing = check_dataset(args.dataset, args.download_dir, raw_assets)

    print(f'Dataset: {args.dataset}')
    print(f'Validation videos expected: {len(ids)}')

    if not missing:
        print('✅ Validation split appears complete.')
        return

    grouped = defaultdict(list)
    for video_id, file_name, base in missing:
        grouped[(video_id, base)].append(file_name)

    print(f'❌ Missing items: {len(missing)}')
    print('First 100 missing entries:')
    shown = 0
    for (video_id, base), names in grouped.items():
        for name in names:
            print(f'  video_id={video_id} file={name} dir={base}')
            shown += 1
            if shown >= 100:
                return


if __name__ == '__main__':
    main()
