from image_processing import *
import datetime
import time

# These environmental variables need to be changed by the user
os.environ['GDAL_DATA'] = '/anaconda3/envs/gis/share/gdal'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/terenceconlon/Google Authentication/qsel-columbia-37d91db96bb8.json'
# fiona.drvsupport.supported_drivers['gml'] = 'rw'
# fiona.drvsupport.supported_drivers['GML'] = 'rw'


def parallel_download(image_list):
    local_image_dir = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles/'

    try:
        download_images_and_cloud_masks(local_image_dir, image_list)

    except Exception as e:
        print(e)



if __name__ == '__main__':

    tile = 'T37NEJ'
    ## These file directories need to be changed by the user.
    local_image_dir = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles/'
    local_utils_dir = '/Volumes/sel_external/sentinel_imagery/image_utils/'

    download_raw_imagery = False
    create_evi_imagery   = False
    stack_imagery        = False
    downsample_imagery   = False
    infill_imagery       = False

    upload_imagery = False



    strt_dt = datetime.date(2017, 1, 1)
    end_dt  = datetime.date(2019, 12, 1)

    t = time.time()

    if download_raw_imagery:
        find_images(local_utils_dir, tile)

        image_list = load_images_within_date_range(local_utils_dir, tile, strt_dt, end_dt)

        print(image_list)
        print(len(image_list))

        for image_tuple in image_list:
            download_images_and_cloud_masks(local_image_dir, image_tuple)

    elapsed = time.time() - t
    print('Elapsed time: {}s'.format(elapsed))

    if create_evi_imagery:
        create_evi_imgs(local_image_dir, tile)

    if stack_imagery:
        stack_images(local_image_dir, tile, 'EVI', strt_dt, end_dt)

    if downsample_imagery:
        in_file = glob.glob(os.path.join(local_image_dir, '{}/stacks/{}_*_10m.tif'.format(tile, tile)))[0]
        ds_factor = 10

        downsample_stack(in_file, ds_factor)

    if infill_imagery:
        in_file = os.path.join(local_image_dir, '{}/stacks/{}_stack_EVI_10m.tif'.format(tile, tile))
        infill_change_dtype(in_file)


    if upload_imagery:

        # In command line, may need to run: gcloud config set core/project qsel-columbia

        bucket_name = 'sentinel_imagery_useast'
        source_file_name = os.path.join(local_image_dir, 'T37PCM/stacks/T37PCM_stack_EVI_10m_infilled.tif')

        # source_file_name = '/Users/terenceconlon/Documents/Columbia - Spring 2020/personal/lebron.jpg'
        # source_file_name = glob.glob('/Volumes/sel_external/sentinel_imagery/reprojected_tiles/{}/stacks/'
        #                              '{}_*_100m_infilled.tif'.format(tile, tile))

        destination_blob_name = os.path.join('ethiopia', source_file_name.split('/')[-1])

        resumable_upload_blob(bucket_name, source_file_name, destination_blob_name)


    # The following lines of code allow for parallel processing
    # You will need to link the threadpool.map call to the function you want to run in parallel
    #

    # parallel = True
    # if parallel:
    #     parallelism = 1  # Try with 4 to see if it crashes; 6 is too high
    #     thread_pool = ThreadPool(parallelism)
    #     thread_pool.map(parallel_download, image_list)
