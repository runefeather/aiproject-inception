from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showerror
from PIL import Image, ImageTk

def load_file():
        # fname = askopenfilename(filetypes=(("Template files", "*.tplate"),
        #                                    ("HTML files", "*.html;*.htm"),
        #                                    ("All files", "*.*") ))
        fname = askopenfilename(filetypes=(("PNG","*.png"),
                                           ("JPEG", "*.jpg;*.jpeg;*.jpe;*jfif"),
                                           ("GIF","*.gif"),
                                           ("TIFF","*.tif;*.tiff"),
                                           ("All files", "*.*") ))
        if fname:
            try:
                print(fname)
                im = Image.open(fname)
                im = im.resize((481,255))
                photo = ImageTk.PhotoImage(im)
                canvas.create_image(10,50, anchor=NW, image=photo)
                canvas.photo = photo
                T.insert(END, str(fname) + " loaded")   
                # filename = PhotoImage(file = "sunshine.gif")
                # canvas.create_image(200, 200, anchor=NE, image=image)
                # canvas.create_rectangle(50, 25, 150, 75, fill="blue")
            except:                     # <- naked except is a bad idea
                # showerror("Open Source File", "Failed to read file\n'%s'" % fname)
                print ("Unexpected error:", sys.exc_info()[1])
            return


window = Tk()
window.title('AI Project')

canvas = Canvas(window, width=500, height=500)
canvas.create_rectangle(10, 50, 490, 300, fill="black")

btnBrowse = Button(canvas, text="Browse", command=load_file, width=10)
btnBrowse.place(x=10,y=10)

T = Text(canvas, height=10, width=68)
T.place(x=10, y=330)
T.insert(INSERT, "Select an image to classify\n")

# btnClassify = Button(canvas, text="Train", command=load_file, width=10)
# btnClassify.place(x=10, y)
canvas.pack()

window.mainloop()

