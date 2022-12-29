import os

import json
from glob import glob

from tqdm import tqdm

def find_id_in_fn_list(fn_list, image_id, extra=None):
  for fn in fn_list:
    if image_id in fn:
      if extra is not None:
        if extra in fn:
          return fn
      else:
        return fn

  raise FileNotFoundError('ID {} not found in fn_list'.format(image_id))


def prepare_data(panoptic_json, output_fn, images_dir, panoptic_dir, semantic_dir, output_json_dir):
  with open(panoptic_json, 'r') as fp:
    panoptic_dict = json.load(fp)

  annotations = panoptic_dict['annotations']

  if not os.path.exists(output_json_dir):
    os.mkdir(output_json_dir)
  images_fn_list = glob(images_dir)
  pan_seg_fn_list = glob(panoptic_dir)
  sem_seg_fn_list = glob(semantic_dir)

  images_per_class = dict()
  for anno in tqdm(annotations):
    cat_ids = list()
    for segm in anno['segments_info']:
      cat_id = segm['category_id']
      if segm['iscrowd'] == 1:
        pass
      else:
        if cat_id not in images_per_class.keys():
          if cat_id not in cat_ids:
            images_per_class[cat_id] = [anno['image_id']]
            cat_ids.append(cat_id)
        else:
          if cat_id not in cat_ids:
            images_per_class[cat_id].append(anno['image_id'])
            cat_ids.append(cat_id)

    segments_info = anno['segments_info']
    image_id = anno['image_id']

    file_name = find_id_in_fn_list(images_fn_list, image_id)
    pan_seg_file_name = find_id_in_fn_list(pan_seg_fn_list, image_id)
    sem_seg_file_name = find_id_in_fn_list(sem_seg_fn_list, image_id, extra='labelTrainIds')

    assert os.path.isfile(file_name), "file_name no path"
    assert os.path.isfile(pan_seg_file_name), "pan_seg no path"
    assert os.path.isfile(sem_seg_file_name), "sem_seg no path"

    out_per_img = {
      "file_name": file_name,
      "image_id": image_id,
      "sem_seg_file_name": sem_seg_file_name,
      "pan_seg_file_name": pan_seg_file_name,
      "segments_info": segments_info,
    }
    json_per_img_fn = os.path.join(output_json_dir, image_id + '.json')
    with open(json_per_img_fn, 'w') as wp:
      json.dump(out_per_img, wp)

    del cat_ids

  for cat_id in images_per_class.keys():
    num_imgs = len(images_per_class[cat_id])
    print("{}: {}".format(cat_id, num_imgs))

  cat_id_list = list(images_per_class.keys())
  output_dict = {'cat_ids': cat_id_list,
                 'imgs_per_cat': images_per_class}

  with open(output_fn, 'w') as wp:
    json.dump(output_dict, wp)


if __name__ == "__main__":
  data_dir = os.getenv("DETECTRON2_DATASETS", "datasets")

  panoptic_json = os.path.join(data_dir, "cityscapes", "gtFine/cityscapes_panoptic_train.json")
  output_fn = os.path.join(data_dir, "cityscapes", "images_per_class_train.json")
  output_json_dir = os.path.join(data_dir, "cityscapes", "panoptic_json_train")

  images_dir = os.path.join(data_dir, "cityscapes", "leftImg8bit/train/*/*")
  panoptic_dir = os.path.join(data_dir, "cityscapes", "gtFine/cityscapes_panoptic_train/*")
  semantic_dir = os.path.join(data_dir, "cityscapes", "gtFine/train/*/*")

  prepare_data(panoptic_json,
               output_fn,
               images_dir,
               panoptic_dir,
               semantic_dir,
               output_json_dir)
