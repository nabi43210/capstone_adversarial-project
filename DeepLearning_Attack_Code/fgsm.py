'''
FGSM_
����:non-targeted
random noise generator
'''

#������Ʈ
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np

#matplot ����
mpl.rcParams['figure.figsize'] = (8, 8) 
mpl.rcParams['axes.grid'] = False


#�����Ʒõ� ��(mobilenetV2)�� class,description,score �� ����
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True)
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

#�𵨿� �°� �̹��� ����
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...] #batch_size, width, height, channel
  return image

def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

#�׽�Ʈ�� �̹��� �ε�
path="/content"
tf.keras.utils.get_file(path+'/testcat.jpg','https://docs.google.com/uc?export=download&id=1oTBz-oEQ0XYf7evpFRNZAHHO_EsA-l2b&confirm=t')
img=tf.io.read_file('/content/testcat.jpg')
img=tf.image.decode_image(img)
image = preprocess(img)
image_probs = pretrained_model.predict(image)
#�׽�Ʈ�� �̹��� �ð�ȭ
plt.figure()
plt.imshow(image[0] * 0.5 + 0.5) 
_, image_class, class_confidence = get_imagenet_label(image_probs)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()


#FGSM ������ (img_descent) 
loss_object = tf.keras.losses.CategoricalCrossentropy()
def img_descent(image, label):
  with tf.GradientTape() as tape:
    tape.watch(image)
    prediction = pretrained_model(image)
    loss = loss_object(label, prediction)

  gradient = tape.gradient(loss, image)
  # sign()���� ��ȣ �ݴ��
  signed_grad = tf.sign(gradient)
  return signed_grad

#������ �ð�ȭ
tiger_cat_index = 282
label = tf.one_hot(tiger_cat_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = img_descent(image, label)
plt.imshow(perturbations[0] * 0.5 + 0.5); 

#���� ����
def display_images(image, description):
  _, label, confidence = get_imagenet_label(pretrained_model.predict(image))
  plt.figure()
  plt.imshow(image[0]*0.5+0.5)
  plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
  plt.show()

epsilons = [0, 0.01, 0.1, 0.2]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]
for i, eps in enumerate(epsilons):
  adv_x = image + eps*perturbations
  adv_x = tf.clip_by_value(adv_x, -1, 1)
  display_images(adv_x, descriptions[i])