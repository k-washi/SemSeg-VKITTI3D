import os
import sys

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(BASE_DIR)

from utils.logger import set_logger
logger = set_logger(__name__)

#----------------------------------------------

import numpy as np


#----------------------------------------------


def _get_filepath_vkitti3d_dataset(dataset_path):
  """
  VKITTK3Dのデータセットのファイル名を取得する
  input: 
    dataset_path: path/to/vkitti3d_dataset_v1.0/*
  output:
    ALL_FILES: ex[..., path/to/vkitti3d_dataset_v1.0/06/0020_00500.npy']
  """
  import glob
  folders = glob.glob(dataset_path) #[... , '/Users/washizakikai/data/vkitti3d_dataset_v1.0/06']
  ALL_FILES = []
  for f in folders:
    ALL_FILES += glob.glob(f + "/*")
  return ALL_FILES

def _load_data(npy_file_path):
  """
  データをロードして格納
  npy_data
  x, y, z, r, g, b, label

  LABEL
    Label ID	Semantics	RGB	Color
  0	Terrain	[200, 90, 0]	brown
  1	Tree	[0, 128, 50]	dark green
  2	Vegetation	[0, 220, 0]	bright green
  3	Building	[255, 0, 0]	red
  4	Road	[100, 100, 100]	dark gray
  5	GuardRail	[200, 200, 200]	bright gray
  6	TrafficSign	[255, 0, 255]	pink
  7	TrafficLight	[255, 255, 0]	yellow
  8	Pole	[128, 0, 255]	violet
  9	Misc	[255, 200, 150]	skin
  10	Truck	[0, 128, 255]	dark blue
  11	Car	[0, 200, 255]	bright blue
  12	Van	[255, 128, 0]	orange
  13	Don't care	[0, 0, 0]	black
  """
  
  f = np.load(npy_file_path) #ex shape (356657, 7)
  data = f[:,:3] #(x, y, z, r, g, b) => {x, y, z}
  label = f[:, -1].astype(np.int) #(label)
  
  return data, label

def _load_all_data(all_data_list, pointNum = 4096):
  """
  すべてのデータを格納する
  """
  t_data = np.empty((len(all_data_list), pointNum, 3))
  t_label = np.empty((len(all_data_list), pointNum))

  for i, f in enumerate(all_data_list):
    data, label = _load_data(f)

    data = np.reshape(data, (1, data.shape[0], data.shape[1]))
    label = np.reshape(label, (1, label.shape[0]))

    idxs = _shuffle_idxs(data.shape[1])
    
    t_data[i, :, :] = data[:, idxs[:pointNum], :]
    t_label[i, :] = label[:, idxs[:pointNum]]
  return t_data, t_label

def _shuffle_idxs(point_num):
  idxs = np.arange(0, point_num)
  np.random.shuffle(idxs)
  return idxs

def load_all_data_vkitti3d(data_path, only=False, pointNum=4096):
  """
  data_pathに対して、すべてのvkitti3dデータセットのデータをロードして、data, labelを返す
  """
  files = _get_filepath_vkitti3d_dataset(data_path)
  if only:
    files = [files[0]]
  t_data, t_label = _load_all_data(files, pointNum=pointNum)
  return t_data, t_label

def data_normalization(total_data):
  """
  正規化したデータと、もとのデータの最小、最大値を返す
  """
  Data_normal = np.zeros(total_data.shape)
  norml_param = np.zeros((3, 2))
  for i in range(3):
    norml_param[i][0] = np.min(total_data[:, :, i])
    norml_param[i][1] = np.max(total_data[:, :, i])
    Data_normal[:, :, i] = (total_data[:, :, i] - norml_param[i][0]) / (norml_param[i][1] - norml_param[i][0])
  return Data_normal, norml_param
    

def _label_unique(total_label):
  #ラベルに含まれる値を取得
  uq_labels = np.unique(total_label)
  logger.debug("ラベルの値: {}".format(uq_labels))

  return uq_labels

def data_colorlize(total_data, total_label):
  uq_labels = _label_unique(total_label)
  


if __name__ == "__main__":
  data_path = "/Users/washizakikai/data/vkitti3d_dataset_v1.0/*"
  logger.debug(BASE_DIR)

  t_data, t_label = load_all_data_vkitti3d(data_path, only=True)
  logger.debug("ReadData shape D:{0}, L:{1}".format(t_data.shape, t_label.shape))

  _label_unique(t_label)
  

  """
  features = ["x","y","z"]
  for i in range(3): 
    print(features[i] + "_range :", np.min(t_data[:, :, i]), np.max(t_data[:, :, i]))

  print("-"*20)
  
  #Normarilzation
  t_data_normal, nm = data_normalization(t_data)
  print(nm)
  for i in range(3): 
    print(features[i] + "_range :", np.min(t_data_normal[:, :, i]), np.max(t_data_normal[:, :, i]))
  """