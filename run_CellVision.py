from multiprocessing import Process
import os
import sys
from main import main
from glob import glob

def run_single(indir, subdir, file_index):
  main(indir, subdir, file_index)

def run(indir, subdir):
  num_devices = int(sys.argv[2]) if len(sys.argv) > 2 else 4
  use_device = int(sys.argv[1]) % num_devices if len(sys.argv) > 1 else -1
  
  if use_device >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(use_device)
  
  files = sorted(glob(os.path.join(indir, subdir, 'reg*.tif')))
  for i, f in enumerate(files):
    if use_device >= 0 and i%num_devices != use_device: continue
    
    print('Processing file: {}, ({} of {})'.format(f,i+1,len(files)))
    p = Process(target=run_single, args=(indir, subdir, i,))
    p.start()
    p.join()


if __name__ == '__main__':
  os.chdir(sys.path[0]) # cd to the CellVision root directory
  
  for d in ['bestFocus']:
     #run('X:/admin/Yuqi/250223_RBT_70194/Scan1', d)
     run('X:/admin/Yuqi/230223_NBT_70129/Scan2', d)
  #for d in ['bestFocus']:
  #  run('V:/admin/John/U54/small_runs/U54_21', d)
  #for d in ['bestFocus']:
  #  run('V:/admin/John/U54/small_runs/U54_22', d)
  #for d in ['bestFocus']:
  #  run('V:/admin/John/U54/small_runs/U54_23', d)
