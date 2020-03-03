import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

from utils.logger import set_logger
logger = set_logger(__name__)

#----------------------------

import open3d as o3d
import numpy as np
import time
#----------------------------

def point_cloud_vis_xyz(data):
  """
  data: (point id, {x, y, z})
  """

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(data)
  o3d.io.write_point_cloud("data/test.ply", pcd)
  o3d.visualization.draw_geometries([pcd])

def point_cloud_vis_all(datas):
  vis = vis_pointcloud()
  vis.create_window()
  #vis.img_capture()
  
  for i in range(datas.shape[0]):
    vis.update(datas[i])
    time.sleep(.25)
    print("Update")
  
  vis.destroy()

class vis_pointcloud(object):
  def __init__(self):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    self.pcd = o3d.geometry.PointCloud()
    self.vis = o3d.visualization.Visualizer()
    
    self.save_img = False
    self.count = 0
  
  def create_window(self):
    self.vis.create_window()
    self.vis.add_geometry(self.pcd)
  
  def img_capture(self):
    self.save_img = True
  
  def update(self, data):
    self.pcd.points = o3d.utility.Vector3dVector(data)
    self.vis.add_geometry(self.pcd)
    
    self.vis.update_geometry()
    self.vis.poll_events()
    self.vis.update_renderer()

    if self.save_img:
      self.vis.capture_screen_image("img/temp_%04d.jpg" %self.count)
    
    self.count += 1
  
  def destroy(self):
    self.vis.destroy_window()

if __name__ == "__main__":
  #test_pcl()
  
  from io_utils.open_data_set import load_all_data_vkitti3d
  data_path = "/Users/washizakikai/data/vkitti3d_dataset_v1.0/*"
  logger.debug(BASE_DIR)

  t_data, t_label = load_all_data_vkitti3d(data_path, pointNum=20000)
  
  point_cloud_vis_xyz(t_data[1])
  #point_cloud_vis_all(t_data)


