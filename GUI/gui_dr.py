from tkinter import * 
from tkinter.filedialog import askopenfilename
from keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

model = model_from_json(open('dr_model.json', 'r').read())
model.load_weights('dr_weights.hdf5')

root = Tk()
root.title('Diabetic Retinopathy Detection')
root.geometry('1000x500')
root.configure(bg='gray')
message = StringVar()
message.set('Select an image and click Predict')

def quit_app():
    root.quit()
    
def file_open():
    global filename
    filename = askopenfilename(title = "Select file", filetypes = (("png files","*.png"),("all files","*.*"),
                                                                  ("jpeg files","*.jpg")))
    
def show_img():
    dr_image = Image.open(filename)
    dr_image = dr_image.resize((300,300), Image.NEAREST)
    dr_image = ImageTk.PhotoImage(dr_image)
    img = Label(root, image=dr_image)
    img.image = dr_image
    img.place(x=50, y=150)
    
def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img
    
def prediction():
    dr_img = cv2.imread(filename)
    dr_img = crop_image_from_gray(dr_img,tol=7)
    dr_img = cv2.resize(dr_img,(224,224))
    dr_img = np.array(dr_img)/255
    dr_img = np.expand_dims(dr_img, axis=0)
    pred = model.predict(dr_img)
    ans = [round(i) for i in pred[0]]
    ans = sum(ans)
    if ans == 1:
        message.set('Person is Healthy')
    elif ans == 2:
        message.set('Person has Mild Diabetic Retinopathy')
    elif ans == 3:
        message.set('Person has Moderate Diabetic Retinopathy')
    elif ans == 4:
        message.set('Person has Severe Diabetic Retinopathy')
    elif ans == 5:
        message.set('Person has Proliferate Diabetic Retinopathy')
    
# Toolbar
toolbar = Frame(root, bd=1)
open_ = Image.open('open.png')
exit_ = Image.open('exit.jpg')

# Importing toolbar images
open_ = open_.resize((35,35), Image.NEAREST)
exit_ = exit_.resize((35,35), Image.NEAREST)
open_icon = ImageTk.PhotoImage(open_)
exit_icon = ImageTk.PhotoImage(exit_)

# Creating Toolbar Buttons
open_button = Button(toolbar, image=open_icon, command=file_open)
exit_button = Button(toolbar, image=exit_icon, command=quit_app)
open_button.image = open_button
exit_button.image = exit_button
open_button.grid(row=0, column=0, sticky='W')
exit_button.grid(row=0, column=1, sticky='W')
toolbar.grid(row=0, column = 0, sticky='W', ipadx=1000)

# Creating Show Image Button
show_img_button = Button(root, text='Show Image', bd=10, bg='light grey', height=2, width=10, command=show_img)
show_img_button.place(x=150, y=50)

# Creating Predict Button
show_img_button = Button(root, text='Predict', bd=10, bg='light grey', height=2, width=10, command=prediction)
show_img_button.place(x=700, y=50)

# Creating label for showing prediction
prediction_label = Label(root, textvariable=message, bg='white', bd=14, height=7, width=40)
prediction_label.place(x=600, y=150)

root.mainloop()