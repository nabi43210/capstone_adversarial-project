import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import fgsm
import jsma
import exif_attack

def on_entry_click(event):
    if entry.get() == "닉네임을 입력하세요":
        entry.delete(0, "end")  # 텍스트 삭제
        entry.config(fg="black")  # 폰트 및 색상 변경

def on_focusout(event):
    if entry.get() == "":
        entry.insert(0, "닉네임을 입력하세요")  # 힌트 텍스트 복원
        entry.config(fg="gray")

window = tk.Tk()
window.title("Protect Your Image")
window.geometry("300x430")

# 이미지를 표시할 레이블
image_label = tk.Label(window)
image_label.pack()

entry = tk.Entry(window, width=30, fg="gray")
entry.insert(0, "닉네임을 입력하세요")  # 힌트 텍스트 설정
entry.bind("<FocusIn>", on_entry_click)  # 포커스 이벤트 처리
entry.bind("<FocusOut>", on_focusout)  # 포커스 이벤트 처리
entry.pack()

# 라디오 박스 2개
selected_option = tk.StringVar()
selected_option.set("option1")

radio_button1 = tk.Radiobutton(window, text="FGSM", variable=selected_option, value="option1")
radio_button1.pack()

radio_button2 = tk.Radiobutton(window, text="JSMA", variable=selected_option, value="option2")
radio_button2.pack()

def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    global filepath
    filepath = file_path
    image = Image.open(file_path)
    image = image.resize((300, 300))  # 이미지 크기 조정 (원하는 크기로 변경)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

open_image_button = tk.Button(window, text="이미지 열기", command=open_image)
open_image_button.pack()

def convert_image():
    selected = selected_option.get()
    
    if selected == "option1":
        print(filepath)
        attack = fgsm.ImageFGSMAttack(filepath)
        result = attack.run_attack()
        save_path = "attack_image.jpg"
        tf.keras.preprocessing.image.save_img(save_path, result[0]*0.5+0.5)
        del attack
        name = entry.get()
        exif_attack.attack(filepath, name)
    elif selected == "option2":
        attack = jsma.ImageJSMAAttack(filepath)
        attack.run_attack()
        del attack
        name = entry.get()
        exif_attack.attack(filepath, name)

convert_button = tk.Button(window, text="변환", command=convert_image)
convert_button.pack()

window.mainloop()
