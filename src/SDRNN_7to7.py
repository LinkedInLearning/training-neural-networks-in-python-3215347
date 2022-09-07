import MLP
from MLP import MultiLayerPerceptron
import tkinter as tk
import numpy as np

sdrnn = MultiLayerPerceptron(layers=[7,7,7])  # MLP
tepochs = 0

def update_a(event):
    r = int(slider_a.get() * (255 - offset)) + offset
    g = offset - int(slider_a.get() * (offset))
    b = offset - int(slider_a.get() * (offset))
    slider_a.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                    troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
    run_ann()

def update_b(event):
    r = int(slider_b.get() * (255 - offset)) + offset
    g = offset - int(slider_b.get() * (offset))
    b = offset - int(slider_b.get() * (offset))
    slider_b.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                    troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
    run_ann()

def update_c(event):
    r = int(slider_c.get() * (255 - offset)) + offset
    g = offset - int(slider_c.get() * (offset))
    b = offset - int(slider_c.get() * (offset))
    slider_c.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                    troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
    run_ann()

def update_d(event):
    r = int(slider_d.get() * (255 - offset)) + offset
    g = offset - int(slider_d.get() * (offset))
    b = offset - int(slider_d.get() * (offset))
    slider_d.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                    troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
    run_ann()

def update_e(event):
    r = int(slider_e.get() * (255 - offset)) + offset
    g = offset - int(slider_e.get() * (offset))
    b = offset - int(slider_e.get() * (offset))
    slider_e.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                    troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
    run_ann()

def update_f(event):
    r = int(slider_f.get() * (255 - offset)) + offset
    g = offset - int(slider_f.get() * (offset))
    b = offset - int(slider_f.get() * (offset))
    slider_f.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                    troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
    run_ann()

def update_g(event):
    r = int(slider_g.get() * (255 - offset)) + offset
    g = offset - int(slider_g.get() * (offset))
    b = offset - int(slider_g.get() * (offset))
    slider_g.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                    troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
    run_ann()

def update_out_a(event):
    r = int(out_a.get() * (255 - offset)) + offset
    g = offset - int(out_a.get() * (offset))
    b = offset - int(out_a.get() * (offset))
    out_a.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                 troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))

def update_out_b(event):
    r = int(out_b.get() * (255 - offset)) + offset
    g = offset - int(out_b.get() * (offset))
    b = offset - int(out_b.get() * (offset))
    out_b.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                 troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))

def update_out_c(event):
    r = int(out_c.get() * (255 - offset)) + offset
    g = offset - int(out_c.get() * (offset))
    b = offset - int(out_c.get() * (offset))
    out_c.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                 troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))

def update_out_d(event):
    r = int(out_d.get() * (255 - offset)) + offset
    g = offset - int(out_d.get() * (offset))
    b = offset - int(out_d.get() * (offset))
    out_d.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                 troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))

def update_out_e(event):
    r = int(out_e.get() * (255 - offset)) + offset
    g = offset - int(out_e.get() * (offset))
    b = offset - int(out_e.get() * (offset))
    out_e.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                 troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))

def update_out_f(event):
    r = int(out_f.get() * (255 - offset)) + offset
    g = offset - int(out_f.get() * (offset))
    b = offset - int(out_f.get() * (offset))
    out_f.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                 troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))

def update_out_g(event):
    r = int(out_g.get() * (255 - offset)) + offset
    g = offset - int(out_g.get() * (offset))
    b = offset - int(out_g.get() * (offset))
    out_g.config(background="#{0:02x}{1:02x}{2:02x}".format(r,g,b),
                 troughcolor="#{0:02x}{1:02x}{2:02x}".format(r,g,b))
                
def default_btn(event):
    train_callback()

def train_callback():
    global tepochs
    epochs = int(entry_epochs.get())
    for i in range(epochs):
        MSE = 0.0
        MSE += sdrnn.bp([1,1,1,1,1,1,0],[1,1,1,1,1,1,0])    #0 pattern
        MSE += sdrnn.bp([0,1,1,0,0,0,0],[0,1,1,0,0,0,0])    #1 pattern
        MSE += sdrnn.bp([1,1,0,1,1,0,1],[1,1,0,1,1,0,1])    #2 pattern
        MSE += sdrnn.bp([1,1,1,1,0,0,1],[1,1,1,1,0,0,1])    #3 pattern
        MSE += sdrnn.bp([0,1,1,0,0,1,1],[0,1,1,0,0,1,1])    #4 pattern
        MSE += sdrnn.bp([1,0,1,1,0,1,1],[1,0,1,1,0,1,1])    #5 pattern
        MSE += sdrnn.bp([1,0,1,1,1,1,1],[1,0,1,1,1,1,1])    #6 pattern
        MSE += sdrnn.bp([1,1,1,0,0,0,0],[1,1,1,0,0,0,0])    #7 pattern
        MSE += sdrnn.bp([1,1,1,1,1,1,1],[1,1,1,1,1,1,1])    #8 pattern
        MSE += sdrnn.bp([1,1,1,1,0,1,1],[1,1,1,1,0,1,1])    #9 pattern
    lbl_err.configure(text="{0:.10f}".format(MSE/10.0))
    tepochs += epochs
    lbl_tepochs.configure(text = tepochs)
    run_ann()

def run_ann():
    x = []
    x.append(slider_a.get())
    x.append(slider_b.get())
    x.append(slider_c.get())
    x.append(slider_d.get())
    x.append(slider_e.get())
    x.append(slider_f.get())
    x.append(slider_g.get())    
    theoutput = sdrnn.run(x)
    out_a.set(theoutput[0])
    out_b.set(theoutput[1])
    out_c.set(theoutput[2])
    out_d.set(theoutput[3])
    out_e.set(theoutput[4])
    out_f.set(theoutput[5])
    out_g.set(theoutput[6])

def start_GUI():
    global offset 
    global root
    global slider_a
    global slider_b
    global slider_c
    global slider_d
    global slider_e
    global slider_f
    global slider_g
    global lbl_err
    global out_a
    global out_b
    global out_c
    global out_d
    global out_e
    global out_f
    global out_g
    global btn_go
    global entry_epochs
    global lbl_tepochs 
    global tepochs
    tepochs = 0
    offset = 240

    # Create the root object
    root = tk.Tk()
    root.configure(background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))
    root.title("7 to 7 SDRNN")
    root.resizable(0,0)
    
    #Create Panedwindow  
    panedwindow = tk.PanedWindow(root, orient='horizontal',
    background="#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))  
    panedwindow.pack(fill="both", expand=True)  
    #Create Frames    
    frame1=tk.Frame(panedwindow,width=100,height=300, relief="sunken", bd=3,
                 background="#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))
    frame2=tk.Frame(panedwindow,width=400,height=400, relief="sunken", bd=3)  
    frame3=tk.Frame(panedwindow,width=100,height=300, relief="sunken", bd=3,
                 background="#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    # Create Scale Widgets
    slider_a = tk.Scale(frame1, orient="horizontal", resolution=0.01,from_=0, to=1,
                            command=update_a, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    slider_b = tk.Scale(frame1, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_b, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    slider_c = tk.Scale(frame1, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_c, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    slider_d = tk.Scale(frame1, orient="horizontal", resolution=0.01,from_=0, to=1,
                            command=update_d, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    slider_e = tk.Scale(frame1, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_e, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    slider_f = tk.Scale(frame1, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_f, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    slider_g = tk.Scale(frame1, orient="horizontal", resolution=0.01,from_=0, to=1,
                            command=update_g, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    # Create Scale Widgets
    out_a = tk.Scale(frame3, orient="horizontal", resolution=0.01,from_=0, to=1,
                            command=update_out_a, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    out_b = tk.Scale(frame3, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_out_b, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    out_c = tk.Scale(frame3, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_out_c, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    out_d = tk.Scale(frame3, orient="horizontal", resolution=0.01,from_=0, to=1,
                            command=update_out_d, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    out_e = tk.Scale(frame3, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_out_e, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    out_f = tk.Scale(frame3, orient="vertical", resolution=0.01,from_=1, to=0,
                            command=update_out_f, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))

    out_g = tk.Scale(frame3, orient="horizontal", resolution=0.01,from_=0, to=1,
                            command=update_out_g, fg="black",width=10, 
                            background = "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset),
                            troughcolor= "#{0:02x}{1:02x}{2:02x}".format(offset,offset,offset))
       
    entry_epochs = tk.Entry(frame2, bd=3, width=10)
    entry_epochs.insert(tk.INSERT, "10")
    entry_epochs.bind('<Return>', default_btn)
    lbl_epochs = tk.Label(frame2, height=2, text="Epochs to Train:")

    lbl_tepochs = tk.Label(frame2, height=2, width=20)
    lbl_tepochs_txt = tk.Label(frame2, height=2, text="Epochs so far:")

    lbl_err = tk.Label(frame2, height=2, width=20)
    lbl_err_txt = tk.Label(frame2, height=2, text="Training Error:")
    
    btn_go = tk.Button(frame2, text="Train some more", command = train_callback)
     
    # Use the grid geometry manager to put the widgets in the respective position
    slider_a.grid(row=0, column=1)
    slider_b.grid(row=1, column=2)
    slider_c.grid(row=3, column=2)
    slider_d.grid(row=4, column=1)
    slider_e.grid(row=3, column=0)
    slider_f.grid(row=1, column=0)
    slider_g.grid(row=2, column=1)

    lbl_epochs.grid(      row=0, column=0)
    entry_epochs.grid(    row=0, column=1)
    btn_go.grid(          row=1, column=1)
    lbl_err.grid(         row=2, column=1)
    lbl_err_txt.grid(     row=2, column=0)
    lbl_tepochs_txt.grid( row=3, column=0)
    lbl_tepochs.grid(     row=3, column=1)

    out_a.grid(row=0, column=1)
    out_b.grid(row=1, column=2)
    out_c.grid(row=3, column=2)
    out_d.grid(row=4, column=1)
    out_e.grid(row=3, column=0)
    out_f.grid(row=1, column=0)
    out_g.grid(row=2, column=1)

    panedwindow.add(frame1)  
    panedwindow.add(frame2) 
    panedwindow.add(frame3) 
    panedwindow.paneconfig(frame1,minsize=200)
    panedwindow.paneconfig(frame2,minsize=250)
    panedwindow.paneconfig(frame3,minsize=200)

    lbl_err.configure(text="---")
    lbl_tepochs.configure(text = "0")
    run_ann()
    # The application mainloop
    tk.mainloop()

start_GUI()