
import glob
import functools
import os
import tensorflow as tf

CITYSCAPES_FOLDER="/content/dataset/cityscapes"
CITYSCAPES_FILE_TEMPLATE=os.path.join('{root}', '{type}', '{split}', '{city}',
      '{city}_{seq:{filler}>{len_fill}}_{frame:{filler}>{len_fill}}_{type}{type2}{ext}')

def get_cityscapes_file_pairs(split='train', city='*', sequence='*', 
                              frame='*', ext='.*', gt_type='labelTrainIds', type='leftImg8bit',
                              root_folder=CITYSCAPES_FOLDER, file_template=CITYSCAPES_FILE_TEMPLATE):
    """
    Fetch pairs of filenames for the Cityscapes dataset.
    Note: wildcards accepted for the parameters (e.g. city='*' to return image pairs from every city)
    :param split:           Name of the split to return pairs from ("train", "val", ...)
    :param city:            Name of the city(ies)
    :param sequence:        Name of the video sequence(s)
    :param frame:           Name of the frame
    :param ext:             File extension
    :param gt_type:         Cityscapes GT type
    :param type:            Cityscapes image type
    :param root_folder:     Cityscapes root folder
    :param file_template:   File template to be applied (default corresponds to Cityscapes original format)
    :return:                List of input files, List of corresponding GT files
    """
    input_file_template = file_template.format(
        root=root_folder, type=type, type2='', len_fill=1, filler='*',
        split=split, city=city, seq=sequence, frame=frame, ext=ext)
    input_files = glob.glob(input_file_template)
    
    gt_file_template = file_template.format(
        root=root_folder, type='gtFine', type2='_'+gt_type, len_fill=1, filler='*',
        split=split, city=city, seq=sequence, frame=frame, ext=ext)
    gt_files = glob.glob(gt_file_template)
    assert(len(input_files) == len(gt_files))
    return sorted(input_files), sorted(gt_files)

def cityscapes_input_fn(split='train', root_folder=CITYSCAPES_FOLDER, resize_to=[256, 256],
                        shuffle=False, batch_size=32, num_epochs=None, augment=False,
                        seed=None,):
  """
    Set up an input data pipeline for semantic segmentation applications on Cityscapes dataset.
    :param split:           Split name ('train', 'val', 'test')
    :param root_folder:     Cityscapes root folder
    :param resize_to:       H x W Dimensions to resize the image and label to
    :param shuffle:         Flag to shuffle the dataset
    :param batch_size:      Batch size
    :param num_epochs:      Number of epochs the dataset would be iterated over
    :param augment:         Flag to augment the pairs
    :param seed:            (opt) Seed

    :return:                tf.data.Dataset
    """
 
  input_files, gt_files = get_cityscapes_file_pairs(split=split, root_folder=root_folder)
  return segmentation_input_fn(input_files, gt_files, shuffle,  seed)
  
def segmentation_input_fn(image_files, gt_files=None, shuffle=False, seed=None):
  """
  Set up an input data pipeline for semantic segmentation applications.
  :param image_files:     List of input image files
  :param gt_files:        (opt.) List of corresponding label image files
  :param resize_to:       H x W Dimensions to resize the image and label to
  :param shuffle:         Flag to shuffle the dataset
  :param batch_size:      Batch size
  :param num_epochs:      Number of epochs the dataset would be iterated over
  :param augment:         Flag to augment the pairs
  :param seed:            (opt) Seed
  :return:                tf.data.Dataset
  """
  # Converting to TF dataset:
  image_files = tf.constant(image_files)
  data_dict = {'image': image_files}
  if gt_files is not None:
      gt_files = tf.constant(gt_files)
      data_dict['segmentation_mask'] = gt_files
    
  dataset = tf.data.Dataset.from_tensor_slices(data_dict)
  if shuffle:
      dataset = dataset.shuffle(buffer_size=1000, seed=seed)
  dataset = dataset.prefetch(1)

  # Batching + adding parsing operation:
  parse_fn = functools.partial(parse_function, resize_to=resize_to, augment=augment)
  dataset = dataset.map(parse_fn, num_parallel_calls=4)
  print(dataset)
  # dataset = dataset.batch(batch_size)

  # dataset = dataset.repeat(num_epochs)

  return dataset

def parse_function(filenames, resize_to=[226, 226], augment=True):
    """
    Parse files into input/label image pair.
    :param filenames:   Dict containing the file(s) (filenames['image'], filenames['label'])
    :param resize_to:   H x W Dimensions to resize the image and label to
    :param augment:     Flag to augment the pair
    :return:            Input tensor, Label tensor
    """
    
    img_filename, gt_filename = filenames['image'], filenames.get('segmentation_mask', None)
    
    # Reading the file and returning its content as bytes:
    image_string = tf.io.read_file(img_filename)
    # Decoding into an image:
    image_decoded = tf.io.decode_jpeg(image_string, channels=3)

    # Converting image to float:
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    # Resizing:
    # image = tf.image.resize(image, resize_to)
  
    if gt_filename is not None:
      # Same for GT image:
      gt_string = tf.io.read_file(gt_filename)
      gt_decoded = tf.io.decode_png(gt_string, channels=1)
      
      gt = tf.cast(gt_decoded, dtype=tf.int32)
      
      # gt = tf.image.resize(gt, resize_to, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      # if augment:
      #       image, gt = _augmentation_fn(image, gt)
    
    return {'image': image, 'segmentation_mask': gt}
