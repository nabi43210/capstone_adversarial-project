'''
jsma- one pixel
'''

#모듈 임포트
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import matplotlib.pyplot as plt

#mobilenetV2 로드(사전훈련된 모델) 및 class와 class description, score를 예측에 저장
pretrained_model = tf.keras.applications.MobileNetV2(include_top=True)
pretrained_model.trainable = False
decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions

#이미지를 모델입력에 맞게 조정
def preprocess(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  image = image[None, ...] 
  return image
def get_imagenet_label(probs):
  return decode_predictions(probs, top=1)[0][0]

#테스트용 이미지 불러오기
path="/content"
tf.keras.utils.get_file(path+'/testcat.jpg','https://docs.google.com/uc?export=download&id=1oTBz-oEQ0XYf7evpFRNZAHHO_EsA-l2b&confirm=t')
img=tf.io.read_file('/content/testcat.jpg')
img=tf.image.decode_image(img)
image = preprocess(img)
image_probs = pretrained_model.predict(image)

#원본이미지 클래스와 예측
_,image_class,class_confidence=get_imagenet_label(image_probs)
print(image_class)
print(class_confidence)

#라벨저장(ONE-HOT으로 변경)
tiger_cat=282
y=tf.one_hot(tiger_cat,1000).reshape([1,1000])
target_lbl=934
targetLbl=tf.one_hot(target_lbl,1000).reshape([1,1000])

def jsmaOnePixel(model, img, lbl, targetLbl, e=28):

    NUM_PIXELS = img.numpy().flatten().shape[0]

    def classProb(img):
        return tf.tensordot(model.predict(img), np.squeeze(targetLbl), axes=1)
    
    imgArr = img.numpy().flatten()
    changedPixels = np.ones([NUM_PIXELS])
    aimg = tf.convert_to_tensor(imgArr.reshape(img.shape), dtype=tf.float32)
    
    with tf.GradientTape(persistent=True) as tape:
        grad = tape.gradient(classProb(aimg), aimg)

    for i in range(e):
        with tf.GradientTape() as g:
            g.watch(aimg)
            class_prob = classProb(aimg)
        if g.gradient(class_prob,aimg) is None:
          continue
        g_grad = g.gradient(class_prob, aimg).numpy().flatten()
        g_grad_masked = np.multiply(g_grad, changedPixels)
        pixelToChange = tf.argmax(g_grad_masked).numpy()
        imgArr[pixelToChange] = 1
        changedPixels[pixelToChange] = 0
        aimg = tf.convert_to_tensor(imgArr.reshape(img.shape), dtype=tf.float32)

    return aimg

#공격
x_adv=jsmaOnePixel(pretrained_model,image,y,targetLbl)
print(x_adv)

#공격이미지 모델에 넣고 예측
def preprocess2(image):
  image = tf.cast(image, tf.float32)
  image = tf.image.resize(image, (224, 224))
  image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
  return image
image3=preprocess2(x_adv)
image3=tf.expand_dims(image3,axis=0)
image3=tf.squeeze(image3,axis=0)
image3_probs=pretrained_model.predict(image3)
_,image2_class,class2_confidence=get_imagenet_label(image3_probs)
print(image2_class)
print(class2_confidence)

plt.figure()
plt.imshow(x_adv[0]*0.5+0.5)
plt.title('{} : {:.2f}% Confidence'.format(image2_class,class2_confidence*100))
plt.show()