import copy
import fiona
import glob
import rasterio
from rasterio.mask import mask
from rasterio import Affine
import subprocess
from multiprocessing.dummy import Pool as ThreadPool
from utils import *

def download_images_and_cloud_masks(local_image_dir, gs_tuple):
    # gs_tuple contains information about the image url, the image size, cloud cover, and the ingestion date

    tile_name = gs_tuple[-1].split('_')[-2]
    # parent_dir = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles'
    tile_folder = os.path.join(local_image_dir, tile_name)

    if not os.path.exists(tile_folder):
        os.mkdir(tile_folder)

    print('Downloading Tile: {}'.format(gs_tuple[-1]))

    # Download BGR + NIR bands for EVI calculation
    valid_bands = ['B02', 'B03', 'B04', 'B08']

    # Find a list of images that are available within the image folder
    storage_client = storage.Client()
    bucket_name = 'gcp-public-data-sentinel-2'

    img_folder_prefix = gs_tuple[-1].split(bucket_name)[-1][1::]
    blobs = storage_client.list_blobs(bucket_name, prefix=img_folder_prefix)
    blobs_names = [j.name for j in blobs]

    # Collect the valid images and cloud cover shape files
    jp2_list = sorted([j for j in blobs_names if '.jp2' in j])
    jp2_band_list = sorted([i for i in jp2_list if i.split('_')[-1][0:3] in valid_bands])
    cloud_cover_gml = [j for j in blobs_names if 'CLOUDS' in j]


    # Extract temporal information
    year = jp2_band_list[0].split('_')[-2][0:4]
    month = jp2_band_list[0].split('_')[-2][4:6]
    day = jp2_band_list[0].split('_')[-2][6:8]

    # Create directories to store the imagery/cloud cover shapefile
    dir_folder, cloud_folder = create_dirs(local_image_dir, tile_name, year, month, valid_bands)

    print('Downloading images')
    for img in jp2_band_list:

        band_dir = img.split('_')[-1].replace('.jp2', '')
        out_filename = img.split('/')[-1].replace('.jp2', '.tif')
        dest = os.path.join(dir_folder, band_dir, out_filename)

        # Download images to corresponding folders.
        # Note: this download process changes the file extension from .jp2 to .tif
        download_blob(bucket_name, img, dest)


    print('Downloading cloud cover')
    if len(cloud_cover_gml) > 0:
        cloud_cover_file = cloud_cover_gml[0]
        cloud_dest = os.path.join(cloud_folder,
                              'cloud_cover_polygons_{}{}{}.shp'.format(year, month, day))

        # Download cloud cover shapefile to corresponding folder.
        # Note: this download process changes the file extension from .gml to .shp
        download_blob(bucket_name, cloud_cover_file, cloud_dest)


def create_evi_imgs(local_image_dir, tile_name):


    nodata_val = 32767

    # Extract folders for the tile
    year_dirs, month_dirs, band_dirs = list_folders(local_image_dir, tile_name)


    for dir in month_dirs:

        # Find images and cloud cover shapefiles in the returned folders
        band_dir_subset = [x for x in band_dirs if dir in x]
        band_dir_img_subset = sorted([x for x in band_dir_subset if x[-3] == 'B'])
        cloud_cover_dir = [x for x in band_dir_subset if x.split('/')[-1] == 'cloud_cover'][0]
        cloud_cover_imgs = glob.glob(cloud_cover_dir + '/*.shp')

        evi_folder_path = os.path.join(dir, 'cropped_evi_stack')
        if not os.path.exists(evi_folder_path):
            os.mkdir(evi_folder_path)

        # Collect lists of the images by band
        band_02_images = sorted(glob.glob(band_dir_img_subset[0] + '/*.tif'))
        band_04_images = sorted(glob.glob(band_dir_img_subset[2] + '/*.tif'))
        band_08_images = sorted(glob.glob(band_dir_img_subset[3] + '/*.tif'))

        # Make sure that there are the correct number of images
        assert len(band_02_images) == len(band_04_images) and len(band_02_images) == len(band_08_images)


        for i in range(len(band_02_images)):

            print('Calculating EVI for {}'.format(band_02_images[i]))
            # Extract image date
            img_date = band_02_images[i].split('_')[-2][0:8]

            # Extract cloud cover shape file
            cloud_cover_shp_name = [j for j in cloud_cover_imgs if img_date in j]

            # Establish out file name
            evi_filename = band_02_images[i].split('/')[-1].replace('B02.tif', 'EVI.tif')
            evi_save_file = os.path.join(evi_folder_path, evi_filename)

            # Read in the images for bands 2,4,8 to create an EVI layer. Crop out the cloud covered pixels
            with rasterio.open(band_02_images[i], 'r') as band2_src:

                evi_meta = copy.copy(band2_src.meta)

                with rasterio.open(band_04_images[i], 'r') as band4_src:
                    with rasterio.open(band_08_images[i], 'r') as band8_src:
                        if len(cloud_cover_shp_name) == 1:

                            print(cloud_cover_shp_name)
                            print('Cloud mask present')

                            # Need to change this for locations where there is no cloud mask
                            try:
                                cc_mask = fiona.open(cloud_cover_shp_name[0])
                                cc_mask_list = [cc_mask[i]['geometry'] for i in range(len(cc_mask))]

                                band_02_cropped, _ = mask(band2_src, cc_mask_list, invert=True, nodata=nodata_val)
                                band_04_cropped, _ = mask(band4_src, cc_mask_list, invert=True, nodata=nodata_val)
                                band_08_cropped, _ = mask(band8_src, cc_mask_list, invert=True, nodata=nodata_val)

                                assert band_02_cropped.shape == band_04_cropped.shape and band_08_cropped.shape ==\
                                    band_08_cropped.shape

                            except Exception as e:
                                print('Empty cloud layer')
                                band_02_cropped = band2_src.read()
                                band_04_cropped = band4_src.read()
                                band_08_cropped = band8_src.read()

                        else:
                            print('No cloud mask present')
                            band_02_cropped = band2_src.read()
                            band_04_cropped = band4_src.read()
                            band_08_cropped = band8_src.read()


            max_zero_values = np.max([np.count_nonzero(band_02_cropped == 0),
                                      np.count_nonzero(band_04_cropped == 0),
                                      np.count_nonzero(band_08_cropped == 0)])

            # Replace the non-valued pixels with np.nan if there are a large portion of them. Allows for interpolation.
            if max_zero_values > (0.05 * band_02_cropped.shape[1] * band_02_cropped.shape[2]):
                print('Replacing bad values for partial images')
                print(img_date)

                band_02_cropped = np.where(band_02_cropped == 0, np.nan, band_02_cropped)
                band_04_cropped = np.where(band_04_cropped == 0, np.nan, band_04_cropped)
                band_08_cropped = np.where(band_08_cropped == 0, np.nan, band_08_cropped)

            # Call the EVI layer creation function
            EVI = convert_to_float_and_evi_func(band_02_cropped, band_04_cropped,
                                                band_08_cropped, nodata_val)

        evi_meta['dtype'] = 'float32'
        evi_meta['nodata'] = 'nan'
        evi_meta['driver'] = 'GTiff'

        # Save the evi layer.
        with rasterio.open(evi_save_file, 'w', **evi_meta) as evi_out:
            evi_out.write(EVI)

def convert_to_float_and_evi_func(band_02_array, band_04_array, band_08_array, nodata_val):
    ## Implementing MODIS-EVI algorithm

    # First convert to floats and scale
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


    return EVI


def stack_images(local_image_dir, tile_name, band_name, strt_dt, end_dt):

    # Find all year, month pairs in the date range
    date_tuples = [(dt.year, dt.month) for dt in rrule(MONTHLY, dtstart=strt_dt, until=end_dt)]

    # Create a folder to hold the stacked images
    median_img_folder = os.path.join(local_image_dir, tile_name)
    stacks_folder = os.path.join(median_img_folder, 'stacks')
    if not os.path.exists(stacks_folder):
        os.mkdir(stacks_folder)

    # Load all the tifs and sort by date in ascending order
    all_tifs = sorted(glob.glob(median_img_folder + '/**/*{}.tif'.format(band_name), recursive=True))

    print('Number of images to stack: {}'.format(len(all_tifs)))

    # Create an out file
    out_file = os.path.join(stacks_folder, '{}_stack_{}_10m.tif'.format(
                tile_name, band_name))

    # Create a empty array to store stacked image
    stacked_evi = np.full((36, 10980, 10980), np.nan, dtype=np.float32)

    for tif in all_tifs:
        with rasterio.open(tif, 'r') as evi_src:
            year = int(tif.split('_')[-2][0:4])
            month = int(tif.split('_')[-2][4:6])

            # Load image and insert at the correct index based on date -- not necessarily the index of the file name
            # in the all_tifs list
            index_list = [i == (year, month) for i in date_tuples]
            index = np.argwhere(index_list)[0][0]

            print(tif)
            print(index)


            stacked_evi[index, :, :] = evi_src.read()

    # Save stacked image
    print('Writing stacked image')
    meta = evi_src.meta.copy()
    meta['count'] = 36
    meta['nodata'] = 'nan'
    with rasterio.open(out_file, 'w', **meta) as out_img:
        out_img.write(stacked_evi)

def downsample_stack(in_file, ds_factor):

    # Downsample stack. Usually only used with the 10m file as input, but this can be changed too
    out_file = in_file.replace('_10m.tif','_{}m.tif'.format(ds_factor*10))

    with rasterio.open(in_file, 'r') as evi_src:
        print('Reading in file')

        evi_array = evi_src.read()[:, 0::ds_factor, 0::ds_factor]
        meta = evi_src.meta

        meta['count'] = 36
        meta['height'] /= ds_factor
        meta['width'] /= ds_factor
        tx = copy.copy(meta['transform'])
        meta['transform'] = Affine(tx[0] * ds_factor, tx[1], tx[2], tx[3], tx[4] * ds_factor, tx[5])

    print('Writing out to: {}'.format(out_file))
    with rasterio.open(out_file, 'w', **meta) as out_img:
        out_img.write(evi_array)

def infill_change_dtype(in_file):


    infilled_stack = in_file.replace('.tif', '_infilled.tif')

    print('Loading {}'.format(in_file))
    with rasterio.open(in_file, 'r') as img_src:
        img_meta = img_src.meta
        img_array = img_src.read()
        print('Infilling')
        # Call infilling function
        img_array = missing_vals_infill(img_array)

    # Change data type to take up less storage
    img_array = img_array.astype(np.uint16)
    img_meta.update({'dtype':'uint16'})
    img_meta.update({'nodata':'0'})

    # Write out infilled file
    print('Writing out to {}'.format(infilled_stack))
    with rasterio.open(infilled_stack, 'w', **img_meta) as img_out:
        img_out.write(img_array)

def missing_vals_infill(stacked_array):

    # For a 10m resolution image, this is the longest step in the process.


    num_imgs = stacked_array.shape[0]

    for ix in range(stacked_array.shape[0]):
        # Find missing indices, where either a 0 or a np.nan exists
        missing_indices = np.logical_or(np.isnan(stacked_array[ix]),  stacked_array[ix] <= 200)

        (missing_x, missing_y) = np.where(missing_indices)

        print('Layer: {}; missing values: {}'.format(ix, np.count_nonzero(missing_indices)))

        for index in range(len(missing_x)):
            row, col = missing_x[index], missing_y[index]
            interp_left = ix
            interp_right = ix

            full_stack = stacked_array[:, row, col]
            # print(full_stack)

            # Interpolate depthwise based on the the nearest pixels (before and after) that have valid data
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

                # Find the data values at the same row, col, but at the correct layers
                interp_left_value = stacked_array[interp_left, row, col]
                interp_right_value = stacked_array[interp_right, row, col]

                # Interpolate
                interp_values = np.interp(range(0, len(indices)), [-1, len(indices)],
                                          [interp_left_value, interp_right_value])

                stacked_array[indices, row, col] = interp_values

            else:
                stacked_array[:, row, col] = 0

    print('\nRemaining missing values: {}'.format(np.count_nonzero(np.isnan(stacked_array))))

    return stacked_array

def parallel_download(image_download_list):

    ## Below is the code to download in parallel

    # parallelism = 1  # Anything above 1 seems to cause errors
    # thread_pool = ThreadPool(parallelism)
    # thread_pool.map(parallel_download, image_list)

    try:
        download_images_and_cloud_masks(image_download_list)

    except Exception as e:
        print('\nCould not download image in parallel')
        print(e)
