import io
import os
import argparse
import matplotlib.pyplot as plt
import PySimpleGUI as sg

sg.theme("DarkTeal2")
from colorizers import *
from PIL import Image
file_types = [("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*")]

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o', '--save_prefix', type=str, default='saved',
                    help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
parser.add_argument('-s', '--save_destination', type=str, default='imgs_out/')
opt = parser.parse_args()


# Define the window's contents
layout = [
        #   [sg.Text("Sử dụng GPU : "),sg.Button('ON',key = 'ON'), sg.Button('OFF',key = 'OFF')],
          [sg.Image(key="-IMAGE-")],
          [sg.Text("Đường dẫn :")],  
          [sg.Input(key='-INPUT-'), sg.FileBrowse('Tìm',key='-INPUT-')],
          [sg.Button("Load Image"), sg.Text('PNG',size=(4,1),key='-PROTO-'), sg.Button("PNG"),sg.Button("JPG")],
          [sg.Text("Đường dẫn lưu : ")],
          [sg.Input(key='-OUTPUTF-'), sg.FolderBrowse('Tìm',key='-OUTPUTF-')],
          [sg.Text(size=(40, 1), key='-OUTPUT-')],
          [sg.Button('Ok'), sg.Button('Thoát')]]

# Create the window
window = sg.Window('Colorization', layout)

# Display and interact with the Window using an Event Loop
fileprox = 'PNG'
while True:
    event, values = window.read()
    
    
    if event == sg.WINDOW_CLOSED or event == 'Thoát':
        break
    elif values['-INPUT-']== '':
        if event == 'JPG' or event == 'PNG' or event=='Load Image' or event =='Ok' :
            window['-OUTPUT-'].update("VUI LONG CHON ANH")
        continue
    else:
        fileimage = values['-INPUT-']
        
        if event == 'Load Image' :
            if os.path.exists(fileimage):
                image = Image.open(values["-INPUT-"])
                image.thumbnail((400,400))
                bio = io.BytesIO()
                image.save(bio, format="PNG")
                window["-IMAGE-"].update(data = bio.getvalue())
                continue
        if event == 'JPG' :
            if os.path.exists(fileimage):
                window['-PROTO-'].update('JPG')
                fileprox = 'JPG'
                continue
        elif event == 'PNG' :
            if os.path.exists(fileimage):
                window['-PROTO-'].update('PNG')
                fileprox = 'PNG'
                continue
            
        filepro = fileprox
        # lấy đường dẫn
        print(filepro)
        filedir = values['-INPUT-']
        filesav = values['-OUTPUTF-']
        # lấy tên file từ đường dẫn
        filename = os.path.basename(filedir)
        # cắt phần tên loại dữ liệu 
        filetxt = os.path.splitext(filename)[0]
        # chuyển thành string
        ten = str(filetxt)
        pro = str(filepro)
        print (filetxt)
        print (filesav)
        #Thông báo xuất ra 
        window['-OUTPUT-'].update('Chạy ' + filename + "!")
        opt.img_path =  values['-INPUT-']
        # See if user wants to quit or window was closed
        # load colorizers
        colorizer_eccv16 = eccv16(pretrained=True).eval()
        colorizer_siggraph17 = siggraph17(pretrained=True).eval()
        if opt.use_gpu:
            colorizer_eccv16.cuda()
            colorizer_siggraph17.cuda()
        
        # default size to process images is 256x256
        # grab L channel in both original ("orig") and resized ("rs") resolutions
        img = load_img(opt.img_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256, 256))
        if opt.use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        # colorizer outputs 256x256 ab map
        # resize and concatenate to original L channel
        # Ảnh đen trắng
        img_bw = postprocess_tens(tens_l_orig, torch.cat((0 * tens_l_orig, 0 * tens_l_orig), dim=1))
        # Ảnh dùng hệ eccv16
        out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
        # Ảnh dùng hệ siggraph17
        out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

        # Lưu file vào img_out với tên được ghép từ tên file với loại thuật toán được sử dụng
        # plt.imsave('imgs_out/%s_eccv16.jpg' % ten, out_img_eccv16)
        # plt.imsave('imgs_out/%s_siggraph17.jpg' % ten, out_img_siggraph17)
        path1 = str(filesav+'/'+ten+'_eccv16.'+pro)
        print(path1)
        path2 = str(filesav+'/'+ten+'_siggraph17.'+pro)
        print(path2)
        path3 = str('imgs_out/'+ten+'_eccv16.'+pro)
        print(path3)
        path4 = str('imgs_out/'+ten+'_siggraph17.'+pro)
        if(filesav!= '') :
            plt.imsave(path1 , out_img_eccv16)
            plt.imsave(path2, out_img_siggraph17)
        else :
            plt.imsave(path3, out_img_eccv16)
            plt.imsave(path4, out_img_siggraph17)
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(img)
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(img_bw)
        plt.title('Input')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        plt.imshow(out_img_eccv16)
        plt.title('Output (ECCV 16)')
        plt.axis('off')

        plt.subplot(2, 2, 4)
        plt.imshow(out_img_siggraph17)
        plt.title('Output (SIGGRAPH 17)')
        plt.axis('off')
        plt.show()
        
window.close()
