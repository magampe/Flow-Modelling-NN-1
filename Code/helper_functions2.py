import os

import fiona
import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.merge
import datetime


def merge_tifs(raster_dir, temp_dir, output_name):
    """
    Merges all tifs within a given directory.
    :param raster_dir: str, path to the directory containing tifs to merge
    :param temp_dir: str, path to the directory of the output merged tif
    :param output_name: str, name of the output merged tif
    """
    rasters = []
    metadata = None
    for tif in raster_dir.glob('*.tif'):
        raster = rasterio.open(tif)
        rasters.append(raster)
        if not metadata:
            metadata = raster.meta.copy()

    mosaic, output = rasterio.merge.merge(rasters)
    metadata.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": output})
    with rasterio.open(temp_dir.joinpath(f'{output_name}.tif'), 'w', **metadata) as file:
        file.write(mosaic)


def reproject_shapefile(shp, temp_dir, output_name, crs='EPSG:6575'):
    """
    Reprojects a shapefile to TN StatePlane (EPSG: 6575).
    :param shp: str, path to the shapefile to be projected
    :param temp_dir: str, path to the directory of the output projected shapefile
    :param output_name: str, name of the output projected shapefile
    :param crs: (optional) str, the EPSG of the projected shapefile
    :return: str, path to projected shapefile
    """
    projected_shapefile = os.path.join(temp_dir, output_name)
    gdf = gpd.read_file(shp)
    gdf = gdf.to_crs(crs)
    gdf.to_file(projected_shapefile)
    return projected_shapefile


def compute_tif_data(shp, raster_dir, temp_dir):
    """
    Clips all rasters within a given directory to the boundary of a shapefile.
    Assumes rasters in raster_dir are named as follows:
        DEM, the Digital Elevation Model raster (EPSG: 4326)
        ASPECT, the Aspect raster (EPSG: 4326)
        ROUGH, the Roughness raster (EPSG: 4326)
        TRI, the Terrain Ruggedness Index raster (EPSG: 4326)
        SLOPE, the projected Slope raster (EPSG: 6575)
        (You can remove these and/or add others)
    :param shp: str, path to the shapefile to be projected
    :param raster_dir: str, path to the directory containing tifs
    :param temp_dir: str, path to the directory of the output tif
    """


    with fiona.open(shp) as f:
        poly = [next(iter(f))['geometry']]
    # Add names of other rasters here
    for i in ('DEM', 'ASPECT', 'ROUGH', 'TRI', 'SLOPE'):
        if i == 'SLOPE':
            projected_shp = reproject_shapefile(shp, temp_dir, 'PROJECTED.shp')
            with fiona.open(projected_shp) as f:
                poly = [next(iter(f))['geometry']]

        with rasterio.open(os.path.join(raster_dir, f'{i}.tif')) as source:
            tif, transform = rasterio.mask.mask(source, poly, crop=True)
            meta = source.meta
            meta.update({"driver": "GTiff", "height": tif.shape[1], "width": tif.shape[2], "transform": transform})

        with rasterio.open(temp_dir, 'w', **meta) as destination:
            destination.write(tif)


def compute_bil_data(shp, raster_dir, temp_dir):
    """
    Clips all .bil rasters within a given directory to the boundary of a shapefile.
    Assumes rasters in raster_dir are named as follows:
        DEM, the Digital Elevation Model raster (EPSG: 4326)
        ASPECT, the Aspect raster (EPSG: 4326)
        ROUGH, the Roughness raster (EPSG: 4326)
        TRI, the Terrain Ruggedness Index raster (EPSG: 4326)
        SLOPE, the projected Slope raster (EPSG: 6575)
        (You can remove these and/or add others)
    :param shp: str, path to the shapefile to be projected
    :param raster_dir: str, path to the directory containing tifs
    :param temp_dir: str, path to the directory of the output tif
    """

    
    # Can convert these to input parameters later
    start_dt = datetime.date(2022, 4, 1)
    end_dt = datetime.date(2022, 6, 30)
    delta = datetime.timedelta(days=1)
    dtype = 'tmean'

    # Create list of filenames to iterate over using this
    file_list = []
    while start_dt <= end_dt:
        dt = start_dt.strftime('%Y%m%d')
        f = f'PRISM_{dtype}_stable_4kmD2_{dt}_bil'
        print(f)
        file_list.append(f)
        start_dt += delta

    # Clip all raster images to the region in the shpfile. Write to a new /tmp/ dir
    with fiona.open(shp) as f:
        poly = [next(iter(f))['geometry']]
    # Add names of other rasters here
    for i in file_list:
        with fiona.open(shp) as f:
            poly = [next(iter(f))['geometry']]
        with rasterio.open(os.path.join(raster_dir, f'{i}'+ '.bil')) as source:
            tif, transform = rasterio.mask.mask(source, poly, crop=True)
            meta = source.meta
            meta.update({"driver": "GTiff", "height": tif.shape[1], "width": tif.shape[2], "transform": transform})

        with rasterio.open(temp_dir + f'{i}'+ '.tif', 'w', **meta) as destination:
            destination.write(tif)
