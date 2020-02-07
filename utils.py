# from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import numpy as np
import pandas as pd
import os
import copy
import fiona
from rasterio.merge import merge
from rasterio.plot import show
import geopandas as gpd
import glob
import zipfile
# from osgeo import gdal
from datetime import date
from multiprocessing.dummy import Pool as ThreadPool
import subprocess
import pickle
import rasterio
from rasterio.mask import mask, raster_geometry_mask
from rasterio.windows import Window
from rasterio import features
# import geojson
from shapely.geometry import shape,mapping, Point, Polygon, MultiPolygon, box
# from utils import splitImageIntoCells, imageStacking, writeImageOut
from pyproj import Proj, transform
from rasterio import Affine
from rasterio.warp import reproject, Resampling
from collections import OrderedDict
import shutil
from rasterio import logging
# from utils import crop_and_resample_lc_map, plotting_pixel_distribution, crop_evi_image, split_images
import datetime
from dateutil.rrule import rrule, MONTHLY
import matplotlib.pyplot as plt


def list_folders(tile_name):

    base_dir = os.path.join('/Volumes/sel_external/sentinel_imagery/reprojected_tiles', tile_name)
    dirs = [x[0] for x in os.walk(base_dir)]

    year_dirs = []
    month_dirs = []
    band_dirs = []

    for dir in dirs:
        dir_level = len(dir.split('/'))

        if dir_level == 7:
            year_dirs.append(dir)
        elif dir_level == 8:
            month_dirs.append(dir)
        elif dir_level == 9:
            band_dirs.append(dir)

    return year_dirs, month_dirs, band_dirs

def create_evi_imgs(tile_name):

    nodata_val = 32767
    year_dirs, month_dirs, band_dirs = list_folders(tile_name)

    for dir in month_dirs:
        year  = int(dir.split('/')[-2])
        month = int(dir.split('/')[-1])

        calculate_evi_stack = True

        if calculate_evi_stack:
            band_dir_subset = [x for x in band_dirs if dir in x]
            band_dir_img_subset = sorted([x for x in band_dir_subset if x[-3] == 'B'])
            cloud_cover_dir = [x for x in band_dir_subset if x.split('/')[-1] == 'cloud_cover'][0]
            cloud_cover_imgs = glob.glob(cloud_cover_dir + '/*.shp')

            evi_folder_path = os.path.join(dir, 'cropped_evi_stack')
            if not os.path.exists(evi_folder_path):
                os.mkdir(evi_folder_path)


            band_02_images = sorted(glob.glob(band_dir_img_subset[0] + '/*.tif'))
            band_03_images = sorted(glob.glob(band_dir_img_subset[1] + '/*.tif'))
            band_04_images = sorted(glob.glob(band_dir_img_subset[2] + '/*.tif'))
            band_08_images = sorted(glob.glob(band_dir_img_subset[3] + '/*.tif'))

            assert len(band_02_images) == len(band_04_images) and len(band_02_images) == len(band_08_images)


            for i in range(len(band_02_images)):

                print('Calculating EVI for {}'.format(band_02_images[i]))

                img_date = band_02_images[i].split('/')[-1][0:10]
                cloud_cover_shp_name = [j for j in cloud_cover_imgs if img_date in j]

                evi_save_file = os.path.join(evi_folder_path, '{}_evi.tif'.format(img_date))
                # if not os.path.exists(evi_save_file):

                with rasterio.open(band_02_images[i], 'r', driver='GTiff') as band2_src:
                    evi_meta = copy.copy(band2_src.meta)
                    rgb_meta = copy.copy(band2_src.meta)

                    with rasterio.open(band_03_images[i], 'r', driver='GTiff') as band3_src:
                        with rasterio.open(band_04_images[i], 'r', driver='GTiff') as band4_src:
                            with rasterio.open(band_08_images[i], 'r', driver='GTiff') as band8_src:
                                if len(cloud_cover_shp_name) == 1:
                                    print('Cloud mask present')
                                    cc_mask = fiona.open(cloud_cover_shp_name[0])
                                    cc_mask_list = [cc_mask[i]['geometry'] for i in range(len(cc_mask))]

                                    band_02_cropped, _ = mask(band2_src, cc_mask_list, invert=True, nodata=nodata_val)
                                    # band_03_cropped, _ = mask(band3_src, cc_mask_list, invert=True, nodata=nodata_val)
                                    band_04_cropped, _ = mask(band4_src, cc_mask_list, invert=True, nodata=nodata_val)
                                    band_08_cropped, _ = mask(band8_src, cc_mask_list, invert=True, nodata=nodata_val)

                                    assert band_02_cropped.shape == band_04_cropped.shape and band_08_cropped.shape ==\
                                        band_08_cropped.shape


                                else:
                                    print('No cloud mask present')
                                    band_02_cropped = band2_src.read()
                                    # band_03_cropped = band3_src.read()
                                    band_04_cropped = band4_src.read()
                                    band_08_cropped = band8_src.read()


                                max_zero_values = np.max([np.count_nonzero(band_02_cropped == 0),
                                                          np.count_nonzero(band_04_cropped == 0),
                                                          np.count_nonzero(band_08_cropped == 0)])


                                if max_zero_values > (0.05 * band_02_cropped.shape[1] * band_02_cropped.shape[2]):
                                    print('Replacing bad values for partial images')
                                    print(img_date)


                                    band_02_cropped = np.where(band_02_cropped == 0, np.nan, band_02_cropped)
                                    # band_03_cropped = np.where(band_03_cropped == 0, np.nan, band_03_cropped)
                                    band_04_cropped = np.where(band_04_cropped == 0, np.nan, band_04_cropped)
                                    band_08_cropped = np.where(band_08_cropped == 0, np.nan, band_08_cropped)

                                EVI = convert_to_float_and_evi_func(band_02_cropped, band_04_cropped,
                                                                    band_08_cropped, nodata_val)

                evi_meta['dtype'] = 'float32'
                evi_meta['nodata'] = 'nan'
                rgb_meta['nodata'] = 'nan'


                with rasterio.open(evi_save_file, 'w', **evi_meta) as evi_out:
                    evi_out.write(EVI)


def convert_to_float_and_evi_func(band_02_array, band_04_array, band_08_array, nodata_val):
    ## Implementing MODIS-EVI algorithm

    # First convert to floats
    band_02_array = np.float32(band_02_array)/10000
    band_04_array = np.float32(band_04_array)/10000
    band_08_array = np.float32(band_08_array)/10000

    # Change to np.nan
    band_02_array[band_02_array == nodata_val/10000] = np.nan
    band_04_array[band_04_array == nodata_val/10000] = np.nan
    band_08_array[band_08_array == nodata_val/10000] = np.nan


    L = 1
    C1 = 6
    C2 = 7.5
    G = 2.5

    numerator = G*(band_08_array - band_04_array)
    denominator = (band_08_array + C1*band_04_array - C2*band_02_array + L)

    EVI = np.divide(numerator, denominator)
    EVI = np.multiply(np.clip(EVI, 0, 1), 10000)

    ## Print statements for sanity checking
    # print(np.count_nonzero(np.isnan(EVI)))
    # print(np.count_nonzero(np.isnan(band_02_array)))
    # print(np.nanmax(EVI))
    # print(np.nanmin(EVI))
    # print(np.nanmean(EVI))
    # print(np.nanmedian(EVI))

    return EVI

def missing_vals_infill(stacked_array):

    num_imgs = stacked_array.shape[0]

    print(np.count_nonzero(np.isnan(stacked_array)))

    for ix in range(stacked_array.shape[0]):
        print(ix)
        missing_indices = np.logical_or(np.isnan(stacked_array[ix]),  stacked_array[ix] <= 200)

        (missing_x, missing_y) = np.where(missing_indices)

        print(np.count_nonzero(missing_indices))

        for index in range(len(missing_x)):
            row, col = missing_x[index], missing_y[index]
            interp_left = ix
            interp_right = ix

            full_stack = stacked_array[:, row, col]
            # print(full_stack)
            if (np.count_nonzero(full_stack == 0) + np.count_nonzero(np.isnan(full_stack))) < 0.7*len(full_stack):

                while np.isnan(stacked_array[interp_left, row, col]) or stacked_array[interp_left, row, col] == 0:
                    interp_left = np.mod(interp_left + num_imgs - 1, num_imgs)

                while np.isnan(stacked_array[interp_right, row, col]) or stacked_array[interp_left, row, col] == 0:
                    interp_right += 1
                    interp_right = np.mod(interp_right, num_imgs)

                if interp_left > interp_right:
                    indices = np.concatenate((np.arange(interp_left+1, num_imgs), np.arange(0, interp_right)))
                else:
                    indices = np.arange(interp_left+1, interp_right)


                interp_left_value = stacked_array[interp_left, row, col]
                interp_right_value = stacked_array[interp_right, row, col]

                # print(interp_left_value, interp_right_value)
                interp_values = np.interp(range(0, len(indices)), [-1, len(indices)],
                                          [interp_left_value, interp_right_value])

                stacked_array[indices, row, col] = interp_values

            else:
                # print('else')
                stacked_array[:, row, col] = 0

    print(np.count_nonzero(np.isnan(stacked_array)))

    return stacked_array


def stack_images(tile_name, band_name):


    strt_dt = datetime.date(2017, 1, 1)
    end_dt = datetime.date(2019, 12, 1)
    date_tuples = [(dt.year, dt.month) for dt in rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]


    median_img_folder = os.path.join('/Volumes/sel_external/sentinel_imagery/reprojected_tiles/',tile_name)
    stacks_folder = os.path.join(median_img_folder, 'stacks')
    if not os.path.exists(stacks_folder):
        os.mkdir(stacks_folder)

    all_tifs = glob.glob(median_img_folder + '/**/*{}.tif'.format(band_name), recursive=True)[6::]

    print(all_tifs)
    print(len(all_tifs))

    tile_template = glob.glob('/Volumes/sel_external/sentinel_imagery/tile_template_tifs/pixel_10m' +
                                 '/*_{}.tif'.format(tile_name))[0]
    with rasterio.open(tile_template, 'r') as template_src:
        template = template_src.read()
        rows = template.shape[1]
        cols = template.shape[2]

    print(rows, cols)

    out_file = os.path.join(stacks_folder, '{}_stack_{}_10m.tif'.format(
                tile_name, band_name))

    stacked_evi = np.full((36, rows, cols), np.nan, dtype=np.float32)

    for tif in all_tifs:
        with rasterio.open(tif, 'r', driver='GTiff') as evi_src:
            year = int(tif.split('/')[-1][0:4])
            month = int(tif.split('/')[-1][5:7])

            index_list = [i == (year, month) for i in date_tuples]
            index = np.argwhere(index_list)[0][0]

            print(tif)
            print(index)

            meta = evi_src.meta.copy()
            meta['count'] = 36
            meta['nodata'] = 'nan'

            stacked_evi[index, :, :] = evi_src.read()
            print(np.count_nonzero(np.isnan(evi_src.read())))
            print(np.count_nonzero(np.isnan(stacked_evi[index, :, :])))


    print('Writing stacked image')

    with rasterio.open(out_file, 'w', **meta) as out_img:
        out_img.write(stacked_evi)


def downsample_stack(tile_name):
    ds = 10

    in_file = glob.glob('/Volumes/sel_external/sentinel_imagery/reprojected_tiles/{}/stacks/{}_*_10m.tif'.format(
        tile_name, tile_name))[0]

    out_file = in_file.replace('_10m.tif','_{}m.tif'.format(ds*10))
    print(out_file)

    # stacked_evi = np.full((36, int(11080 / ds), int(10980 / ds)), np.nan, dtype=np.float32)


    with rasterio.open(in_file, 'r', driver='GTiff') as evi_src:
        print('Reading')

        evi_array = evi_src.read()[:, 0::ds, 0::ds]
        meta = evi_src.meta

        print(meta)

        meta['count'] = 36
        meta['height'] /= ds
        meta['width'] /= ds
        tx = copy.copy(meta['transform'])
        meta['transform'] = Affine(tx[0] * ds, tx[1], tx[2], tx[3], tx[4] * ds, tx[5])

    print('Writing')
    with rasterio.open(out_file, 'w', **meta) as out_img:
        out_img.write(evi_array)

def infill_change_dtype(tile_name, band_name):

    pathname = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles/{}/stacks/{}_stack_{}_10m.tif'.format(
        tile_name, tile_name, band_name)

    print(pathname)
    infilled_stack = pathname.replace('_10m.tif', '_infilled_10m.tif')
    print(infilled_stack)


    with rasterio.open(pathname, 'r') as img_src:
        img_meta = img_src.meta
        print('loading')
        img_array = img_src.read()
        print('infilling')
        img_array = missing_vals_infill(img_array)

    img_array = img_array.astype(np.uint16)
    img_meta.update({'dtype':'uint16'})
    img_meta.update({'nodata':'0'})

    with rasterio.open(infilled_stack, 'w', **img_meta) as img_out:
        img_out.write(img_array)


def trim_index_csv():
    index_csv = pd.read_csv('/Volumes/Conlon Backup 2TB/GCP Sentinel Hosting/index.csv')
    index_csv_usa = index_csv[(index_csv['NORTH_LAT'] <= 15) &
                              (index_csv['SOUTH_LAT'] >= 3) &
                              (index_csv['WEST_LON'] >= 32) &
                              (index_csv['EAST_LON'] <= 48) &
                              (index_csv['GEOMETRIC_QUALITY_FLAG'] != 'FAILED') &
                              (index_csv['CLOUD_COVER'] <= 30)]
    index_csv_usa.to_csv('/Volumes/Conlon Backup 2TB/GCP Sentinel Hosting/index_eth_only.csv')

