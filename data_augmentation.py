import tensorflow as tf

class data_augmentation():

  # def rotate45(input_image,input_mask): the addon don't work
  #   input_image=tfa.image.rotate(input_image, tf.constant(np.pi/8))
  #   input_mask=tfa.image.rotate(input_mask, tf.constant(np.pi/8))
  #   return  input_image,input_mask

  def rotate90(input_image,input_mask):
    
    input_image=tf.image.rot90(input_image)
    input_mask=tf.image.rot90(input_mask)
    return input_image,input_mask
  def flip_left_right(input_image,input_mask):
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)
    return input_image,input_mask
  def adjust_saturation(input_image,input_mask):
    random_number=tf.random.uniform([],minval=2,maxval=4,seed=SEED)
    input_image = tf.image.adjust_saturation(input_image, random_number)
    input_mask=input_mask #only can apply saturation when you have a image in 3 channel
    return input_image,input_mask
  def adjust_brightness(input_image,input_mask):
    random_number=tf.random.uniform([],minval=0,maxval=1,seed=SEED)
    input_image=tf.image.adjust_brightness(input_image, 0.4)
    return input_image,input_mask
  def central_crop(input_image,input_mask):
    random_number=tf.random.uniform([],minval=0.5,maxval=1,seed=SEED)
    input_image=tf.image.central_crop(input_image, central_fraction=0.75)
    input_mask=tf.image.central_crop(input_mask,central_fraction=0.75)
    return input_image,input_mask
  
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1  
  return input_image, input_mask
  
def do_technique_of_data_augmentation(input_image,input_mask,list_of_data_augmentation_techniques,PROBABLITY_THRESHOLD):
  for technique,is_used in list_of_data_augmentation_techniques.items():
    function_name=str(technique)[4:].lower()
    if tf.random.uniform(()) > PROBABLITY_THRESHOLD and is_used:
      function=getattr(data_augmentation,function_name)
      input_image,input_mask=function(input_image,input_mask)
  return input_image,input_mask