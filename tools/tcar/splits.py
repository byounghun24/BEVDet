# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Dict, List

from nuscenes import NuScenes

train = ['n298', 'n228', 'n039', 'n321', 'n051', 'n076', 'n336', 'n311', 'n027', 'n181', 'n238', 'n067', \
        'n355', 'n262', 'n054', 'n266', 'n290', 'n204', 'n171', 'n308', 'n032', 'n073', 'n211', 'n219', 'n329', 'n161', \
        'n210', 'n077', 'n191', 'n246', 'n011', 'n066', 'n134', 'n244', 'n128', 'n065', 'n248', 'n267', 'n007', 'n214', \
        'n353', 'n254', 'n110', 'n070', 'n095', 'n075', 'n008', 'n160', 'n038', 'n185', 'n179', 'n409', 'n270', 'n344', \
        'n200', 'n340', 'n094', 'n090', 'n272', 'n356', 'n193', 'n334', 'n101', 'n150', 'n229', 'n100', 'n297', 'n098', \
        'n349', 'n163', 'n216', 'n218', 'n124', 'n375', 'n279', 'n391', 'n221', 'n357', 'n388', 'n287', 'n421', 'n164', \
        'n231', 'n114', 'n020', 'n159', 'n119', 'n314', 'n295', 'n168', 'n386', 'n417', 'n320', 'n174', 'n376', \
        'n208', 'n028', 'n301', 'n104', 'n222', 'n026', 'n400', 'n352', 'n106', 'n247', 'n091', 'n381', 'n335', 'n412', 'n294', \
        'n049', 'n105', 'n178', 'n063', 'n232', 'n148', 'n147', 'n319', 'n306', 'n227', 'n241', 'n393', 'n401', 'n061', 'n132', 'n370', \
        'n085', 'n009', 'n239', 'n201', 'n406', 'n316', 'n343', 'n019', 'n273', 'n293', 'n313', 'n282', 'n142', 'n277', 'n365', 'n087', \
        'n323', 'n285', 'n368', 'n031', 'n116', 'n133', 'n390', 'n096', 'n337', 'n172', 'n103', 'n382', 'n183', 'n351', 'n374', 'n346', \
        'n397', 'n330', 'n268', 'n315', 'n156', 'n224', 'n373', 'n068', 'n043', 'n419', 'n122', 'n169', 'n338', 'n289', 'n197', 'n394', \
        'n331', 'n018', 'n209', 'n125', 'n188', 'n022', 'n021', 'n243', 'n146', 'n255', 'n250', 'n015', 'n062', 'n264', 'n207', 'n213', \
        'n395', 'n225', 'n304', 'n278', 'n010', 'n251', 'n166', 'n001', 'n083', 'n192', 'n292', 'n398', 'n269', 'n153', 'n055', 'n260', 'n092', 'n257', 'n305', 'n002', 'n348', \
        'n081', 'n223', 'n364', 'n422', 'n384', 'n341', 'n369', 'n059', 'n006', 'n129', 'n271', 'n240', 'n196', 'n198', 'n033', 'n385', 'n387', 'n079', 'n057', 'n025', 'n047', \
        'n253', 'n261', 'n071', 'n327', 'n361', 'n205', 'n299', 'n220', 'n300', 'n135', 'n276', 'n399', 'n127', 'n416', 'n372', 'n074', 'n235', 'n203', 'n256', 'n318', 'n363', \
        'n291', 'n109', 'n034', 'n138', 'n206', 'n162', 'n407', 'n118', 'n029', 'n167', 'n392', 'n286', 'n423', 'n139', 'n350', 'n237', 'n345', 'n418', 'n274', 'n088', 'n413', \
        'n312', 'n037', 'n332', 'n137', 'n108', 'n182', 'n190', 'n084', 'n187', 'n326', 'n233', 'n383', 'n377', 'n052', 'n403', 'n366', 'n149', 'n117', 'n339', 'n024', 'n036', \
        'n099', 'n296', 'n186', 'n317', 'n322', 'n151', 'n283', 'n041', 'n194', 'n064', 'n275', 'n236', 'n023', 'n136', 'n310', 'n177', 'n184', 'n050', 'n195', 'n405', 'n414', \
        'n173', 'n111', 'n080', 'n389', 'n175', 'n408', 'n358', 'n082', 'n004', 'n143', 'n302', 'n230', 'n113', 'n215', 'n411', 'n360', 'n333', 'n367', 'n102', 'n288', 'n014', \
        'n309', 'n259', 'n120', 'n112', 'n048', 'n016', 'n017', 'n217', 'n303', 'n045', 'n280', 'n420', 'n347', 'n053', 'n378', 'n072', 'n115', 'n126', 'n141', 'n380', 'n013', \
        'n058', 'n328']

val = ['n212', 'n012', 'n144', 'n252', 'n249', 'n396', 'n044', 'n003', 'n281', 'n089', 'n404', 'n324', 'n097', 'n245', 'n170', 'n307', 'n107', 'n046', 'n131', 'n145', 'n415', \
       'n242', 'n354', 'n056', 'n258', 'n342', 'n152', 'n263', 'n410', 'n202', 'n042', 'n226', 'n158', 'n284', 'n176', 'n199', 'n154', 'n359', 'n069', 'n180', 'n379', 'n325', \
        'n402', 'n189', 'n093', 'n086', 'n078', 'n371', 'n121', 'n362', 'n234', 'n165', 'n035', 'n140', 'n265', 'n123', 'n040', 'n030', 'n130', 'n005', 'n155', 'n157', 'n060']

test = \
['n018', 'n019', 'n020', 'n021', 'n022', 'n023', 'n024', 'n025', 'n026', 'n027']

mini_train = \
    ['scene-0061', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-1077', 'scene-1094', 'scene-1100']

mini_val = \
    ['scene-0103', 'scene-0916']


def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
