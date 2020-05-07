from image_processing import *
import datetime


os.environ['GDAL_DATA'] = '/anaconda3/envs/gis/share/gdal'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/terenceconlon/Google Authentication/qsel-columbia-37d91db96bb8.json'
fiona.drvsupport.supported_drivers['gml'] = 'rw'
fiona.drvsupport.supported_drivers['GML'] = 'rw'



if __name__ == '__main__':
    tile = 'T37PCM'

    download_raw_imagery = False
    create_evi_imagery   = False
    stack_imagery        = False
    downsample_imagery   = False
    infill_imagery       = False

    upload_imagery = True



    strt_dt = datetime.date(2017, 1, 1)
    end_dt = datetime.date(2019, 12, 1)


    if download_raw_imagery:
        find_images(tile)

        image_list = load_images_within_date_range(tile, strt_dt, end_dt)

        print(image_list)
        print(len(image_list))

        for image_tuple in image_list:
            download_images_and_cloud_masks(image_tuple)


    if create_evi_imagery:
        create_evi_imgs(tile)

    if stack_imagery:
        stack_images(tile, 'EVI', strt_dt, end_dt)

    if downsample_imagery:
        in_file = glob.glob('/Volumes/sel_external/sentinel_imagery/reprojected_tiles/{}/stacks/{}_*_10m.tif'.format(
            tile, tile))[0]
        ds_factor = 10

        downsample_stack(in_file, ds_factor)

    if infill_imagery:
        in_file = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles/{}/stacks/' \
                  '{}_stack_EVI_10m.tif'.format(tile, tile)
        infill_change_dtype(in_file)


    if upload_imagery:

        # In command line, may need to run: gcloud config set core/project qsel-columbia

        bucket_name = 'sentinel_imagery_useast'
        source_file_name = '/Volumes/sel_external/sentinel_imagery/reprojected_tiles/T37PCM/stacks/' \
                           'T37PCM_stack_EVI_100m_infilled.tif'
        destination_blob_name = 'ethiopia/T37PCM_stack_EVI_100m_infilled.tif'
        # destination_blob_name = source_file_name.split('reprojected_tiles')[-1]

        resumable_upload_blob(bucket_name, source_file_name, destination_blob_name)
