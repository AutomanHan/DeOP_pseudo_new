import argparse
import os
import os.path as osp
import shutil
from functools import partial
from glob import glob

import mmcv
import numpy as np
from PIL import Image

COCO_LEN = 123287

full_clsID_to_trID = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    30: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
    92: 80,
    93: 81,
    94: 82,
    95: 83,
    96: 84,
    97: 85,
    98: 86,
    99: 87,
    100: 88,
    101: 89,
    102: 90,
    103: 91,
    104: 92,
    105: 93,
    106: 94,
    107: 95,
    108: 96,
    109: 97,
    110: 98,
    111: 99,
    112: 100,
    113: 101,
    114: 102,
    115: 103,
    116: 104,
    117: 105,
    118: 106,
    119: 107,
    120: 108,
    121: 109,
    122: 110,
    123: 111,
    124: 112,
    125: 113,
    126: 114,
    127: 115,
    128: 116,
    129: 117,
    130: 118,
    131: 119,
    132: 120,
    133: 121,
    134: 122,
    135: 123,
    136: 124,
    137: 125,
    138: 126,
    139: 127,
    140: 128,
    141: 129,
    142: 130,
    143: 131,
    144: 132,
    145: 133,
    146: 134,
    147: 135,
    148: 136,
    149: 137,
    151: 138,
    150: 139,
    152: 140,
    153: 141,
    154: 142,
    155: 143,
    156: 144,
    157: 145,
    158: 146,
    159: 147,
    160: 148,
    161: 149,
    162: 150,
    163: 151,
    164: 152,
    165: 153,
    166: 154,
    167: 155,
    168: 156,
    169: 157,
    170: 158,
    171: 159,
    172: 160,
    173: 161,
    174: 162,
    175: 163,
    176: 164,
    177: 165,
    178: 166,
    179: 167,
    180: 168,
    181: 169,
    182: 170,
    255: 255,
}

novel_clsID = [20, 24, 32, 33, 40, 56, 86, 99, 105, 123, 144, 147, 148, 168, 171]
base_clsID = [k for k in full_clsID_to_trID.keys() if k not in novel_clsID + [255]]
novel_clsID_to_trID = {k: i for i, k in enumerate(novel_clsID)}
base_clsID_to_trID = {k: i for i, k in enumerate(base_clsID)}


def convert_to_trainID(
    maskpath, out_mask_dir, is_train, clsID_to_trID=full_clsID_to_trID, suffix=""
):
    mask = np.array(Image.open(maskpath))
    mask_copy = np.ones_like(mask, dtype=np.uint8) * 255
    for clsID, trID in clsID_to_trID.items():
        mask_copy[mask == clsID] = trID
    seg_filename = (
        osp.join(out_mask_dir, "train2017" + suffix, osp.basename(maskpath))
        if is_train
        else osp.join(out_mask_dir, "val2017" + suffix, osp.basename(maskpath))
    )
    if len(np.unique(mask_copy)) == 1 and np.unique(mask_copy)[0] == 255:
        return
    Image.fromarray(mask_copy).save(seg_filename, "PNG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert COCO Stuff 164k annotations to mmsegmentation format"
    )  # noqa
    parser.add_argument("coco_path", help="coco stuff path")
    parser.add_argument("-o", "--out_dir", help="output path")
    parser.add_argument("--nproc", default=16, type=int, help="number of process")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    coco_path = args.coco_path
    nproc = args.nproc
    print(full_clsID_to_trID)
    print(base_clsID_to_trID)
    print(novel_clsID_to_trID)

    out_dir = args.out_dir or coco_path
    out_mask_dir = osp.join(out_dir, "stuffthingmaps_detectron2")
    for dir_name in [
        "train2017",
        # "test2017",
        "val2017",
        "train2017_base",
        "train2017_novel",
        "val2017_base",
        "val2017_novel",
    ]:
        os.makedirs(osp.join(out_mask_dir, dir_name), exist_ok=True)
    train_list = glob(osp.join(coco_path, "stuffthingmaps", "train2017", "*.png"))
    test_list = glob(osp.join(coco_path, "stuffthingmaps", "val2017", "*.png"))
    assert (
        len(train_list) + len(test_list)
    ) == COCO_LEN, "Wrong length of list {} & {}".format(
        len(train_list), len(test_list)
    )

    if args.nproc > 1:
        mmcv.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
            nproc=nproc,
        )

        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
            nproc=nproc,
        )
        mmcv.track_parallel_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
            nproc=nproc,
        )
    else:
        mmcv.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=True),
            train_list,
        )
        mmcv.track_progress(
            partial(convert_to_trainID, out_mask_dir=out_mask_dir, is_train=False),
            test_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            train_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=base_clsID_to_trID,
                suffix="_base",
            ),
            test_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=True,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            train_list,
        )
        mmcv.track_progress(
            partial(
                convert_to_trainID,
                out_mask_dir=out_mask_dir,
                is_train=False,
                clsID_to_trID=novel_clsID_to_trID,
                suffix="_novel",
            ),
            test_list,
        )
    print("Done!")


if __name__ == "__main__":
    main()
