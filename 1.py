import cv2
import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import sys

config_file=r'SR-DTU\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model=r'SR-DTU\frozen_inference_graph.pb'

app=customtkinter.CTk()
app.title('Human detection')
app.geometry(f"{1100}x{580}")
app.resizable(False,False)
customtkinter.set_appearance_mode('system')
customtkinter.set_appearance_mode('green')

model = cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels=['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant',
        'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
        'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
        'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon',
        'bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
        'pottedplant','bed','diningtable','toilet','tvmonitorlaptop','mouse','remote','keyboard','cell phone','microwave',
        'oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

model.setInputSize(320,320)
model.setInputScale(1/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN

cap=cv2.VideoCapture(0)
def main():
    def baack():
        global cont
        #os._exit
        cap.release()
        cv2.destroyAllWindows()
        app.destroy()
        sys.exit()

    labelFramerr = customtkinter.CTkFrame(master=app,width=800,height=500,corner_radius=15)
    labelFramerr.place(relx=0.5,rely=0.5,anchor=tkinter.CENTER)

    headingLabelrr = customtkinter.CTkLabel(master=labelFramerr, text="Human detection", font=('Century Gothic',20))
    headingLabelrr.place(x=320,y=40)

    quitBtnrr = customtkinter.CTkButton(master=labelFramerr,text="Quit",width=180,command=baack,corner_radius=6)
    quitBtnrr.place(x=320,y=450)
    while True:
        ret,frame = cap.read()

        classIndex, confidece, bbox = model.detect(frame,confThreshold=0.6)

        if (len(classIndex)!=0):
            for ClassInd, conf, boxes in zip(classIndex.flatten(),confidece.flatten(), bbox):
                if (ClassInd==1):
                    cv2.rectangle(frame,boxes,(255,0,0),2)
                    cv2.putText(frame,classLabels[ClassInd-1]+str(confidece*100),(boxes[0]+10,boxes[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        #cv2.imshow('Detection in progress',frame)

        img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=ImageTk.PhotoImage(Image.fromarray(img1))
        width=1000
        height=700
        dim=(width,height)
        resized=cv2.resize(img1,dim,interpolation=cv2.INTER_LINEAR)
        img=ImageTk.PhotoImage(Image.fromarray(resized))
        l1=customtkinter.CTkLabel(master=labelFramerr,height=300,width=700,text='',image=img)
        l1.place(relx=0.5,rely=0.51,anchor='center')
        app.update()

main()