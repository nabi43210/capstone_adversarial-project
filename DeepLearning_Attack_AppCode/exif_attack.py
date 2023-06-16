import piexif

def attack(image_path, name):
    javascript_code = "alert('Please... Obey the copyright!!!');"
    exif_dict = piexif.load(image_path)
    exif_dict["0th"][piexif.ImageIFD.Software] = javascript_code
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)
    copyright_info = "Copyright © " + name
    exif_dict["0th"][piexif.ImageIFD.Artist] = copyright_info
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)