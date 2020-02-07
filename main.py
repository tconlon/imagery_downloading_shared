# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import numpy as np
import pandas as pd
import os
import fiona
import glob
from multiprocessing.dummy import Pool as ThreadPool
import subprocess
import pickle
import rasterio
from rasterio.warp import reproject, Resampling
from utils import trim_index_csv
from retrying import retry

import datetime
from dateutil.rrule import rrule, MONTHLY


# @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000)

os.environ['GDAL_DATA'] = '/anaconda3/envs/gis/share/gdal'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/terenceconlon/Google Authentication/elec-vis-8f629093a79a.json'
fiona.drvsupport.supported_drivers['gml'] = 'rw'
fiona.drvsupport.supported_drivers['GML'] = 'rw'


def find_images(tile):
    index_csv = pd.read_csv('/Volumes/Conlon Backup 2TB/GCP Sentinel Hosting/index_eth_only.csv')

    gs_storage_list = []
    save_file = os.path.join('/Volumes/Conlon Backup 2TB/sentinel_imagery/gcp_sentinel_imagery_utils/image_lists',
                                    'image_lists_by_tile', 'ethiopia', 'valid_tiles_T{}.pkl'.format(tile))


    tile_dict = {}


    valid_images = index_csv.loc[(index_csv['MGRS_TILE'] == tile) ]

    for index, row in valid_images.iterrows():
        if row['GRANULE_ID'][0:3] != 'L1C':
            valid_images = valid_images.drop(index)

    # Create dict
    for i in range(len(valid_images)):
        sensing_time = valid_images['SENSING_TIME'].iloc[i].split('-')
        year, month = sensing_time[0:2]
        day = sensing_time[2][0:2]

        ym_tuple = (int(year), int(month))
        image_info =  (valid_images['CLOUD_COVER'].iloc[i], year, month, day, valid_images['BASE_URL'].iloc[i])

        if ym_tuple not in tile_dict.keys():
            tile_dict[ym_tuple] = [image_info]
        else:
            tile_dict[ym_tuple].append(image_info)

    # Sort dict
    for key in tile_dict.keys():
        tile_dict[key] = sorted(tile_dict[key], key= lambda x: (x[0], np.abs(int(x[3])-15)))



    with open(save_file, 'wb') as f:
        pickle.dump(tile_dict, f)


def load_images_within_date_range(tile):
    strt_dt = datetime.date(2016, 9, 1)
    end_dt = datetime.date(2019, 8, 1)
    date_tuples = [(dt.year, dt.month) for dt in rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]

    trimmed_dict = {}
    image_list = []

    corrupted_images = [(2019,3)]


    save_file = os.path.join('/Volumes/Conlon Backup 2TB/sentinel_imagery/gcp_sentinel_imagery_utils/image_lists',
                             'image_lists_by_tile', 'ethiopia', 'valid_tiles_T{}.pkl'.format(tile))

    with open(save_file, 'rb') as f:
        tile_dict = pickle.load(f)

    for key in tile_dict.keys():

        try:
            if key in date_tuples:
                if key in corrupted_images:
                    image_list.append(tile_dict[key][1])
                else:
                    image_list.append(tile_dict[key][0])

        except Exception as e:
            print(e)
            print('Image not available for {}'.format(key))

    return image_list


def create_dirs(save_dir_base, tile_name, year, month, valid_bands):

    folder_level_1 = os.path.join(save_dir_base, tile_name)
    folder_level_2 = os.path.join(folder_level_1, year)
    folder_level_3 = os.path.join(folder_level_2, month)
    folder_level_cloud = os.path.join(folder_level_3, 'cloud_cover')

    for band in valid_bands:
        folder_level_band = os.path.join(folder_level_3, band)

    if not os.path.isdir(folder_level_1):
        os.mkdir(folder_level_1)
    if not os.path.isdir(folder_level_2):
        os.mkdir(folder_level_2)
    if not os.path.isdir(folder_level_3):
        os.mkdir(folder_level_3)
    if not os.path.exists(folder_level_cloud):
        os.mkdir(folder_level_cloud)
    for band in valid_bands:
        folder_level_band = os.path.join(folder_level_3, band)
        if not os.path.isdir(folder_level_band):
            os.mkdir(folder_level_band)

    return folder_level_3, folder_level_cloud


def reproject_and_save_tile(gs_tuple, pixel_size = 10):

    tile_name = gs_tuple[-1].split('_')[-2]
    tile_folder_base = '/Volumes/Conlon Backup 2TB/sentinel_imagery/reprojected_tiles'
    tile_folder = os.path.join(tile_folder_base, tile_name)

    if not os.path.exists(tile_folder):
        os.mkdir(tile_folder)

    print('Downloading Tile: {}'.format(gs_tuple[-1]))

    valid_bands = ['B02', 'B03', 'B04', 'B08']

    cmd_string = '/Users/terenceconlon/anaconda3/envs/gis/bin/gsutil ls -r ' + gs_tuple[-1] +'/GRANULE'
    image_list_output = subprocess.Popen(cmd_string, shell= True, stdout=subprocess.PIPE)
    image_list_clean = [j.decode('utf-8') for j in image_list_output.stdout.readlines()]

    image_list_output.kill()

    jp2_list      = sorted([j.replace('\n', '') for j in image_list_clean if '.jp2' in j])
    jp2_band_list = sorted([i for i in jp2_list if i.split('_')[-1][0:3] in valid_bands])
    jp2_single_image = [j.replace('\n', '') for j in image_list_clean if valid_bands[0]+'.jp2' in j][0]
    cloud_cover_gml = [j.replace('\n', '') for j in image_list_clean if 'CLOUDS' in j][0]

    year  = jp2_single_image.split('_')[-2][0:4]
    month = jp2_single_image.split('_')[-2][4:6]
    day   = jp2_single_image.split('_')[-2][6:8]

    dir_folder, cloud_folder = create_dirs(tile_folder_base, tile_name, year, month, valid_bands)


    cloud_path = os.path.join(cloud_folder,
                              'cloud_cover_polygons_{}_{}_{}.shp'.format(year, month, day))

    tile_template = '/Volumes/Conlon Backup 2TB/sentinel_imagery/tile_template_tifs/' \
                    'pixel_{}m/template_{}.tif'.format(pixel_size, tile_name)

    with rasterio.open(tile_template, 'r') as band_dest:
        metadata = band_dest.meta.copy()


        for file in jp2_band_list:
            band = file.split('_')[-1][0:3]
            band_folder = os.path.join(dir_folder, band)
            previously_saved_images = glob.glob(band_folder + '/*.tif')

            save_file_str = '{}_{}_{}_{}.tif'.format(year, month, day, band)
            save_file = os.path.join(band_folder, save_file_str)
            save_new_file = True

            print(save_file)
            if os.path.exists(save_file):
                with rasterio.open(save_file, 'r', driver='GTiff') as saved_img:
                    print(np.max(saved_img.read()))
                    if np.max(saved_img.read()) > 0:
                        save_new_file = False

            print(save_new_file)
            if save_new_file:

                with rasterio.open(file, 'r', driver='JP2OpenJPEG',) as band_src:

                    print('before reproject: {}'.format(file))

                    with rasterio.open(save_file, 'w', **metadata) as dest_jp2:
                        reproject(source=rasterio.band(band_src,1),
                                    destination=rasterio.band(dest_jp2,1),

                                  resampling=Resampling.nearest)

                    print('after reproject')

    cloud_file = os.path.join(tile_folder, cloud_path)

    if not os.path.exists(cloud_file):
        with fiona.open(cloud_cover_gml, 'r') as src_cc:
            with fiona.open(cloud_file, 'w', crs=src_cc.crs, driver='ESRI Shapefile',
                                schema=src_cc.schema) as output:
                for f in src_cc:
                    output.write(f)



def parallel_download(args):
    gs_storage_list = args

    try:
        reproject_and_save_tile(gs_storage_list)

    except Exception as e:
        print('\nCould not download for image: {}'.format(args[-1]))
        print(e)



if __name__ == '__main__':
    tile = '37PCN'
    parallel = True
    find_new_images = False

    if find_new_images:
        find_images(tile)

    image_list = load_images_within_date_range(tile)

    print(image_list)

    #
    # if parallel:
    #     parallelism = 1  # Try with 4 to see if it crashes; 6 is too high
    #     thread_pool = ThreadPool(parallelism)
    #     thread_pool.map(parallel_download, image_list)