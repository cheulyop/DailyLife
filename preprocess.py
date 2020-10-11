import os
import json
import logging
import argparse
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from utils import chauvenet
from utils.logging import init_logger


def load_data(data_dir, uid):
    '''Search through data_dir until a datafile for the user with given uid is found, and load the data.'''
    for fname in os.listdir(data_dir):
        if uid == int(fname.split('.')[0]):
            fpath = os.path.join(data_dir, fname)
            break

    else:
        raise FileNotFoundError(f'Unable to find data for user {uid:04d}.')

    return pd.read_pickle(fpath, compression='gzip')


def raw_to_segments(paths, configs, logger):
    # get esm data and make dict from uid to esms
    esm_data = pd.read_csv(filepath_or_buffer=paths['esm_path'], header=0)
    uid_to_esms = {uid: esm_data.loc[esm_data['uid'] == uid] for uid in esm_data.uid.unique()}

    # get config variables
    ws, aug, norm = configs['ws'], configs['aug'], configs['norm']
    logger.info(f'Config: size={ws}s, augment={aug}, normalize={norm}')

    # make dataset name
    dataset_name = ('aug' if aug else '') + ('norm' if norm else '') + (f'_{ws}s' if aug or norm else f'{ws}s')
    logger.info(f'Extracting segments to {os.path.join(paths["save_dir"], dataset_name)}...')

    # for each user and corresponding esms
    num_segs = 0
    for uid, esms in uid_to_esms.items():
        try:
            # try loading datafile
            data = load_data(paths['data_dir'], uid)[['Gsr-Resistance', 'HeartRate-Quality', 'HeartRate-BPM', 'SkinTemperature-Temperature', 'RRInterval-Interval']]

            # create directory to save processed segments if it's not there already
            save_path = os.path.join(paths['save_dir'], dataset_name, f'{uid:04d}')
            os.makedirs(save_path, exist_ok=True)
        
        # if datafile for current user does not exist, print error msg and skip
        except FileNotFoundError as err:
            logger.error(err)
            continue
        
        # set 'HeartRate-BPM' value to nan where 'HeartRate-Quality' = 'ACQUIRING'
        data['HeartRate-BPM'].mask(data['HeartRate-Quality'] == 'ACQUIRING', inplace=True)

        # drop 'HeartRate-Quality' column
        data.drop(columns=['HeartRate-Quality'], inplace=True)

        # rename columns names for simplicity
        data.rename(
            columns={
                'Gsr-Resistance': 'gsr',
                'HeartRate-BPM': 'bpm',
                'SkinTemperature-Temperature': 'temp',
                'RRInterval-Interval': 'rri'
            },
            inplace=True
        )

        # remove outliers using chauvenet's criterion (do we want this?)
        data.mask(chauvenet.criterion(data), inplace=True)

        # remove too stable consecutive values (currently 3s) from gsr
        gsr = data.gsr.dropna()
        data['gsr'].mask(gsr.groupby(gsr.diff().ne(0).cumsum()).transform('size').ge(5*3), inplace=True)

        # apply z-score normalization if norm = True
        if norm:
            data = pd.DataFrame(StandardScaler().fit_transform(data), index=data.index, columns=data.columns)
        
        # for each esm
        for esm in tqdm(esms.itertuples(), total=len(esms), desc=f'User {uid}', ascii=True, dynamic_ncols=True):
            # get segment: from start (response_ts - segment length) to end (response_ts)
            end = esm.response_ts
            start = end - (1e3 * ws)  # ws is in seconds
            seg = data.loc[lambda x: (x.index >= start) & (x.index < end + (1e3 * 30))]  # add 30 seconds to end to avoid akward cutoffs
            
            # if none of columns are entirely empty
            if seg.notnull().sum().all():
                # resample and interpolate signals with respective sampling rates
                seg.index = pd.DatetimeIndex(seg.index * 1e6)
                gsr = seg.gsr.dropna().resample('200ms').mean().interpolate(method='time')[:ws * 5]  # fs = 5Hz
                bpm = seg.bpm.dropna().resample('1S').mean().interpolate(method='time')[:ws]  # fs = 1Hz
                rri = seg.rri.dropna().resample('1S').mean().interpolate(method='time')[:ws]  # fs = 1Hz
                temp = seg.temp.dropna().resample('30S').mean().interpolate(method='time')[:ws // 30]  # fs = 30s

                # make sure that number of not-nan values in current segment is as expected
                try:
                    assert gsr.count() == ws * 5 and bpm.count() == ws and rri.count() == ws and temp.count() == ws // 30
                except AssertionError:
                    # otherwise print warning message and skip
                    logger.warning(f'Signal length mismatch: gsr={gsr.count()}, bpm={bpm.count()}, rri={rri.count()}, temp={temp.count()}')
                    continue
                
                # save current segment as json file
                sig = {
                    'gsr': gsr.tolist(),
                    'bpm': bpm.tolist(),
                    'rri': rri.tolist(),
                    'temp': temp.tolist(),
                }
                with open(os.path.join(save_path, f'{esm.Index:04d}.json'), 'w') as f:
                    json.dump(sig, f)
                num_segs += 0

    return num_segs


if __name__ == "__main__":
    # init parser
    parser = argparse.ArgumentParser(description='Process DailyLife dataset and save biosignal segments as JSON files.')
    parser.add_argument('--root', '-r', type=str, required=True, help='a path to a root directory for the dataset')
    parser.add_argument('--timezone', '-t', type=str, default='UTC', help='a pytz timezone string for logger, default is UTC')
    parser.add_argument('--size', '-s', type=int, default=60, help='segment size in seconds')
    parser.add_argument('--augment', default=False, action='store_true', help='augment with duration if set to True, default is False')
    parser.add_argument('--normalize', default=False, action='store_true', help='apply Z-normalization if set to True, default is False')
    args = parser.parse_args()

    # check commandline arguments
    assert args.size >= 60, f'Segment size must be greater than or equal to 60, but given {args.size}.'

    # init default logger
    logger = init_logger(tz=args.timezone)
    logger.info(f'Read/writing files to {os.path.expanduser(args.root)}...')

    # paths to load and save data
    PATHS = {
        'esm_path': os.path.expanduser(os.path.join(args.root, 'metadata/esms_activity.csv')),
        'data_dir': os.path.expanduser(os.path.join(args.root, 'aggregated/')),
        'save_dir': os.path.expanduser(os.path.join(args.root, 'datasets/')),
    }

    # configuration variables for preprocessing
    CONFIGS = {
        'ws': args.size,
        'aug': args.augment,
        'norm': args.normalize,
    }

    num_segs = raw_to_segments(PATHS, CONFIGS, logger)
    logger.info(f'Preprocessing complete, extracted {num_segs} segments.')
