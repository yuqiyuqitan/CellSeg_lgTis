# main.py
# ---------------------------
# main.py connects segmentation, stitching, and output into a single pipeline.  It prints metadata about
# the run, and then initializes a segmenter and stitcher.  Looping over all image files in the directory,
# each image is segmented, stitched, grown, and overlaps resolved.  The data is concatenated if outputting
# as quantifications, and outputted per file for other output methods.  This file can be run by itself by
# invoking python main.py or the main function imported.

import os
from src.cvsegmenter import CVSegmenter
from src.cvstitch_plane import CVMaskStitcher
from src.cvmask import CVMask
from src import cvutils
from src import cvvisualize
from src.my_fcswrite import write_fcs
from cvconfig import CVConfig
from PIL import Image
import skimage
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict

import matplotlib.pyplot as plt
from timeit import default_timer as timer
from time import sleep

def show(img):
  fig = plt.figure()
  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)
  ax.imshow(img, aspect='equal')
  plt.show()

def main(indir, subdir, region_index=None, increase_factor=None, growth_plane=None, growth_quant_A=None, growth_quant_M=None, border_quant_M=None):
  print('Starting CellVision')
  
  physical_devices = tf.config.list_physical_devices('GPU') 
  try:
    for dev in physical_devices:
      tf.config.experimental.set_memory_growth(dev, True) 
  except: # Invalid device or cannot modify virtual devices once initialized. 
    pass
  
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
  
  cf = CVConfig(indir, subdir, increase_factor, growth_plane, growth_quant_A, growth_quant_M, border_quant_M)
  
  print('Initializing CVSegmenter at', cf.DIRECTORY_PATH)
  if cf.IS_CODEX_OUTPUT:
    print('Picking channel', cf.NUCLEAR_CHANNEL_NAME, 'from', len(cf.CHANNEL_NAMES), 'total to segment on')
    print('Channel names:')
    print(cf.CHANNEL_NAMES)
  
  stitcher = CVMaskStitcher(overlap=cf.OVERLAP, min_area=cf.MIN_AREA)
  
  if cf.OUTPUT_METHOD not in ['imagej_text_file', 'statistics', 'visual_image_output', 'visual_overlay_output', 'all']:
    raise NameError('Output method is not supported.  Check the OUTPUT_METHOD variable in cvconfig.py.')
  
  growth = '_us{:.1f}_grow{:.1f}x{:.1f}x{:.1f}b{:.1f}'.format(cf.INCREASE_FACTOR, cf.GROWTH_PIXELS_MASKS, cf.GROWTH_PIXELS_PLANE, cf.GROWTH_PIXELS_QUANT_M, cf.BORDER_PIXELS_QUANT_M)
  
  if region_index is not None and cf.FILENAMES:
    cf.FILENAMES = [cf.FILENAMES[region_index]] if len(cf.FILENAMES) > region_index else []
  
  for count, filename in enumerate(cf.FILENAMES):
    t0_file = t0 = timer()
    
    print('Processing image: {}'.format(filename))
    
    path = os.path.join(cf.DIRECTORY_PATH, filename)
    
    image = np.array(cf.READ_METHOD(path))
    
    print('Load image: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    print('Initial image shape: {}'.format(image.shape))
    print('Image dtype: {}'.format(image.dtype))
    
    if cf.IS_CODEX_OUTPUT and image.ndim == 4:
      if image.shape[3] in [1,2,3,4] and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0, 3)) # cycles, height, width, channels
      else:
        image = np.transpose(image, (2, 3, 0, 1)) # cycles, channels, height, width
      image = image.reshape(image.shape[:2] + (-1,))
    if cf.IS_CODEX_OUTPUT and image.ndim == 3:
      if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        image = np.transpose(image, (1, 2, 0)) # channels,height,width => height, width, channels
    cf.SHAPE = image.shape
    
    nuclear_index = None
    if cf.N_DIMS == 3 or cf.N_DIMS == 4:
      nuclear_index = cvutils.get_channel_index(cf.NUCLEAR_CHANNEL_NAME, cf.CHANNEL_NAMES)
    
    print('Image shape: {}'.format(image.shape))
    print('Using channel {} as nuclear image'.format(nuclear_index+1))
    
    nuclear_image = cvutils.get_nuclear_image(image, nuclear_index=nuclear_index)
    nuclear_image = cvutils.boost_image(nuclear_image, cf.BOOST)
    
    h, w = nuclear_image.shape[:2]
    
    rundir = os.path.basename(os.path.normpath(indir))
    cache_dir = 'X:/admin/Yuqi/CellVisionSegmenter_Colin_v2_old_for_Fusion/CellVisionSegmenter_Colin_v2/cache'
    data_cache_file = '{}/{}_{}_{}x{}_n{}_if{:.1f}_ma{}_gm{}.npz'.format(cache_dir,rundir,subdir,filename,h,w,nuclear_index,cf.INCREASE_FACTOR,cf.MIN_AREA,cf.GROWTH_PIXELS_MASKS)
    
    if os.path.exists(data_cache_file):
      with np.load(data_cache_file) as data_cached:
        mask = data_cached['data']
    else:
      print('\nSegmenting with CellVision:', filename); t0=timer()
      
      segmenter = CVSegmenter(
        cf.SHAPE,
        cf.MODEL_PATH,
        cf.OVERLAP,
        cf.INCREASE_FACTOR,
        cf.MIN_AREA
      )
      
      rois, masks, scores, rows, cols = segmenter.segment_image(nuclear_image)
      
      del segmenter.model
      del segmenter
      
      print('Detect cells: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      if cf.GROWTH_PIXELS_MASKS:
        cvutils.dilate_masks(rois, masks, scores, h, w, rows, cols, cf.OVERLAP, cf.GROWTH_PIXELS_MASKS)
      
      mask = stitcher.stitch_masks_plane(rois, masks, scores, rows, cols, h, w)
      np.savez_compressed(data_cache_file, data=mask)
    
    stitched_mask = CVMask(mask)
    
    n = stitched_mask.n_instances
    print(n, 'cell masks found by segmenter')
    if n == 0:
      print('No cells found in', filename)
      return
    
    if cf.GROWTH_PIXELS_PLANE:
      print('Growing cells by {} pixels'.format(cf.GROWTH_PIXELS_PLANE))
      stitched_mask.greydilate(cf.GROWTH_PIXELS_PLANE)
      print('Grow plane mask: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    print('Computing cell centroids and ROIs')
    t0 = timer()
    stitched_mask.compute_centroids()
    print('Compute centroids and ROIs: {:.1f}s'.format(timer()-t0)); t0=timer()
    
    if not os.path.exists(cf.IMAGEJ_OUTPUT_PATH): os.makedirs(cf.IMAGEJ_OUTPUT_PATH)
    if not os.path.exists(cf.VISUAL_OUTPUT_PATH): os.makedirs(cf.VISUAL_OUTPUT_PATH)
    if not os.path.exists(cf.QUANTIFICATION_OUTPUT_PATH): os.makedirs(cf.QUANTIFICATION_OUTPUT_PATH)
    
    outname = filename[:-4] if filename.endswith('.tif') else filename
    if os.path.isfile(os.path.join(indir, 'experiment.json')):
      with open(os.path.join(indir, 'experiment.json')) as json_file:
        run_name = json.load(json_file)['name']
        outname = '{}_reg{:03d}'.format(run_name, region_index+1)
    
    if cf.OUTPUT_METHOD == 'imagej_text_file' or cf.OUTPUT_METHOD == 'all' and False:
      print('Saving coordinates for ImageJ:', filename)
      print('I have not verified that this output is correct.  Check it before running this')
      seriously_i_have_not_checked_the_results_at_all
      
      new_path = os.path.join(cf.IMAGEJ_OUTPUT_PATH, (outname + '-coords.txt'))
      stitched_mask.output_to_file(new_path)
    
    if cf.OUTPUT_METHOD == 'visual_image_output' or cf.OUTPUT_METHOD == 'all':
      print('Saving image output to', cf.VISUAL_OUTPUT_PATH)
      visual_path = os.path.join(cf.VISUAL_OUTPUT_PATH, outname) + '_visual' + growth
      Image.fromarray(stitched_mask.plane_mask).save(visual_path + '_labeled.png')
      cvvisualize.save_mask_overlays(visual_path, nuclear_image, stitched_mask.plane_mask, stitched_mask.rois)
    
    stitched_region = 'mosaic' in filename or 'stitched' in filename
    
    if n > 0:
      if cf.OUTPUT_METHOD == 'statistics' or cf.OUTPUT_METHOD == 'all':
        print('Calculating statistics')
        reg, tile_row, tile_col, tile_z = 0, 1, 1, 0
        if cf.IS_CODEX_OUTPUT:
          if stitched_region:
            reg, tile_z = cvutils.extract_stitched_information(filename)
          else:
            reg, tile_row, tile_col, tile_z = cvutils.extract_tile_information(filename)
      
      centroids = stitched_mask.centroids
      xys = np.fliplr(np.around(centroids))
      
      if cf.output_adjacency_quant:
        t0 = timer()
        
        areas, means_u, means_c = stitched_mask.quantify_channels_adjacency(image, cf.GROWTH_PIXELS_QUANT_A, grow_neighbors=True, normalize=True)
        
        print('Quantify cells across channels: {:.1f}s'.format(timer()-t0)); t0=timer()
        
        metadata_list = np.array([reg, 1])
        metadata = np.column_stack([np.arange(1,1+n), np.broadcast_to(metadata_list, (n, len(metadata_list)))])
        zs = np.full([n,1], tile_z)
        
        data_u   = np.concatenate([metadata, xys, zs, xys, areas[:,None], means_u], axis=1)
        data_c   = np.concatenate([metadata, xys, zs, xys, areas[:,None], means_c], axis=1)
        
        labels = [
          'cell_id:cell_id',
          'region:region',
          'tile_num:tile_num',
          'x:x',
          'y:y',
          'z:z',
          'x_tile:x_tile',
          'y_tile:y_tile',
          'size:size'
        ]
        
        # Output to .csv
        ch_names = ['cyc{:03d}_ch{:03d}:{}'.format((i//4)+1,(i%4)+1,s) for i,s in enumerate(cf.CHANNEL_NAMES)]
        cols = labels + ch_names
        
        pd.DataFrame(data_u, columns=cols).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'uncompensated', outname + '_uncompensated.csv'), index=False)
        pd.DataFrame(data_c, columns=cols).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH,   'compensated', outname +   '_compensated.csv'), index=False)

        # Output to .fcs
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'uncompensated', outname + '_uncompensated.fcs'), data_u, cols, split=':')
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH,   'compensated', outname +   '_compensated.fcs'), data_c, cols, split=':')
        
        print('Save measurements to csv: {:.1f}s'.format(timer()-t0)); t0=timer()
      
      if cf.output_morphological_quant:
        t0 = timer()
        
        QL, QT = stitched_mask.quantify_channels_morphological(image, cf.GROWTH_PIXELS_QUANT_M, cf.BORDER_PIXELS_QUANT_M)
        
        print('Quantify cells across channels: {:.1f}s'.format(timer()-t0)); t0=timer()
        
        metadata_list = np.array([reg, 1])
        metadata = np.column_stack([np.arange(1,1+n), np.broadcast_to(metadata_list, (n, len(metadata_list)))])
        zs = np.full([n,1], tile_z)
        
        AL, ML = QL
        data_L_fib = np.concatenate([metadata, xys, zs, xys, AL[0][:,None], AL[1][:,None], AL[2][:,None], ML[0], ML[1], ML[2]], axis=1)
        data_L_f   = np.concatenate([metadata, xys, zs, xys, AL[0][:,None], ML[0]], axis=1)
        
        AT, MT = QT
        data_T_fib = np.concatenate([metadata, xys, zs, xys, AT[0][:,None], AT[1][:,None], AT[2][:,None], MT[0], MT[1], MT[2]], axis=1)
        data_T_f   = np.concatenate([metadata, xys, zs, xys, AT[0][:,None], MT[0]], axis=1)
        
        labels = [
          'cell_id:cell_id',
          'region:region',
          'tile_num:tile_num',
          'x:x',
          'y:y',
          'z:z',
          'x_tile:x_tile',
          'y_tile:y_tile',
          'size:size'
        ]
        
        # Output to .csv
        ch_names = ['cyc{:03d}_ch{:03d}:{}'.format((i//4)+1,(i%4)+1,s) for i,s in enumerate(cf.CHANNEL_NAMES)]
        cols_fib = labels + ['interior:interior', 'border:border'] + ch_names + [s + '_interior' for s in ch_names] + [s + '_border' for s in ch_names]
        cols_f   = labels + ch_names
        
        pd.DataFrame(data_L_fib, columns=cols_fib).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_fib' + growth + '_loose.csv'), index=False)
        pd.DataFrame(data_T_fib, columns=cols_fib).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_fib' + growth + '_tight.csv'), index=False)
        
        pd.DataFrame(data_L_f  , columns=cols_f  ).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_loose.csv'), index=False)
        pd.DataFrame(data_T_f  , columns=cols_f  ).to_csv(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_tight.csv'), index=False)
        
        # Output to .fcs file
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_fib' + growth + '_loose.fcs'), data_L_fib, cols_fib, split=':')
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_fib' + growth + '_tight.fcs'), data_T_fib, cols_fib, split=':')
        
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'loose', outname + '_loose.fcs'), data_L_f, cols_f, split=':')
        write_fcs(os.path.join(cf.QUANTIFICATION_OUTPUT_PATH, 'tight', outname + '_tight.fcs'), data_T_f, cols_f, split=':')
      
      print('Total processing time for file {}: {:.1f}m'.format(filename, (timer()-t0_file) / 60));

if __name__ == "__main__":
  main()
