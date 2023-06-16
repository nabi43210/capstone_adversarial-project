'''
FGSM_
유형:non-targeted
random noise generator
'''

#모델임포트
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np

#matplot 설정
mpl.rcParams['figure.figsize'] = (8, 8) 
mpl.rcParams['axes.grid'] = False


class ImageFGSMAttack:
    def __init__(self, filepath):
        #사전훈련된 모델(mobilenetV2)와 class,description,score 을 저장
        self.pretrained_model = tf.keras.applications.MobileNetV2(include_top=True)
        self.pretrained_model.trainable = False
        self.decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions
        self.filepath = filepath
    
    #모델에 맞게 이미지 조정
    def preprocess(self, image):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (224, 224))
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = image[None, ...] #batch_size, width, height, channel
        return image

    def get_imagenet_label(self, probs):
        return self.decode_predictions(probs, top=1)[0][0]

    def run_attack(self):
        #테스트용 이미지 로드
        img=tf.io.read_file(self.filepath)
        img=tf.image.decode_image(img)
        image = self.preprocess(img)
        image_probs = self.pretrained_model.predict(image)

        #FGSM 생성기 (img_descent) 
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        def img_descent(image, label):
            with tf.GradientTape() as tape:
                tape.watch(image)
                prediction = self.pretrained_model(image)
                loss = loss_object(label, prediction)

            gradient = tape.gradient(loss, image)
            # sign()으로 부호 반대로
            signed_grad = tf.sign(gradient)
            return signed_grad

        #노이즈 시각화
        tiger_cat_index = 282
        label = tf.one_hot(tiger_cat_index, image_probs.shape[-1])
        label = tf.reshape(label, (1, image_probs.shape[-1]))
        perturbations = img_descent(image, label)
        #plt.imshow(perturbations[0] * 0.5 + 0.5); 
        #공격 적용
        def display_images(image, description):
            _, label, confidence = self.get_imagenet_label(self.pretrained_model.predict(image))
            plt.figure()
            plt.imshow(image[0]*0.5+0.5)
            plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
            plt.show()
        result = None
        epsilons = [0, 0.01, 0.1, 0.2]
        descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input') for eps in epsilons]
        for i, eps in enumerate(epsilons):
            adv_x = image + eps*perturbations
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            if i == len(epsilons) - 1:
                result = adv_x
        #display_images(adv_x, descriptions[-1])
        return result