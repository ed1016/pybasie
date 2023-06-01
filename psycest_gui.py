
# gui libraries
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle
from scipy.optimize import minimize_scalar
from scipy.stats import norm, logistic, norm
from scipy.special import ndtri, expit, erf, erfinv, erfc, erfcinv
from datetime import datetime as dt
import math
import random
import glob
import pandas as pd
from pydub import AudioSegment
from pydub.playback import play

# own libraries
from basie_functions import *
from basie_class import *
# from basie_functions import v_quadpeak


def defaultfct(**kwargs):
    print('this is the defaultfct')
def play_audio(**kwargs):
    audiofile=kwargs.get('audiofile')
    audiovar=kwargs.get('audioVar')
    if not audiovar.get():
        print('playing this file: ', audiofile)
        song = AudioSegment.from_wav(audiofile)
        play(song);
        audiovar.set('True')
    else:
        print('already played audio')
def extractfilelist(folderpath):
    foldfiles = os.listdir(folderpath)

    files = []
    availsnr = []
    reverblist = []
    for filename in foldfiles:
        reverblist.append((filename[filename.find("reverb_")+len("reverb_"):filename.find("_snr_")]))
    reverblist=np.unique(reverblist)

    for i in range(len(reverblist)):
        tmpfiles = []
        tmpavailsnr = []
        for filename in foldfiles:
            if filename.endswith(".wav") and ("practice" not in filename) and (reverblist[i] in filename):
                tmpfiles.append(filename)
                tmpavailsnr.append(float(filename[filename.find("snr_")+len("snr_"):filename.find("_db")]))
        files.append(tmpfiles)
        availsnr.append(tmpavailsnr)

    return reverblist, files, availsnr
def extractfilelist_practice(folderpath):
    foldfiles = os.listdir(folderpath)
    files = []
    availsnr = []
    for filename in foldfiles:
        if filename.endswith(".wav") and "practice" in filename:
            files.append(filename)
            availsnr.append(float(filename[filename.find("snr_")+len("snr_"):filename.find("_db")]))
    return files, availsnr
def check_ID_day(outfolder, ID):
    foldfiles = os.listdir(outfolder)
    timestamp = dt.now().strftime("%Y%m%d")
    matches=[]
    for i in foldfiles:
        if i==ID:
            timefiles=os.listdir(os.path.join(outfolder,i))
            for j in timefiles:
                if timestamp in j:
                    matches.append(os.path.join(outfolder,i, j))
    return matches


def run_practice(**kwargs):
    nt = int(kwargs.get('ntrials'))
    audiofiles = kwargs.get('audiofiles').get()
    root = kwargs.get('rootfig')
    method = kwargs.get('method').get()
    methodfiles=kwargs.get('methodfiles').get().split(',')

    filelist, availsnr = extractfilelist_practice(audiofiles)
    if not filelist :
        print('No .wav files in provided folder')
    elif not availsnr:
        print('Wrong snr naming convention in provided folder')
    else:

        snrlist = list(set(availsnr))
        print(snrlist)
        filenames=np.empty((nt, 1), dtype=object)
        for i in range(int(nt)):
            flg=0
            while flg==0:
                snr = snrlist[i]
                snridx = np.where(np.in1d(availsnr, snr))[0];
                currentfile=random.choice([filelist[j] for j in snridx])
                filenames[i,:] = currentfile
                trialtitle = 'Practice ' + str(i+1) +'/' + str(nt)
                if method=='MRT [Hurr.]':
                    response, wordchoice = responsewindow_hurricane(root, os.path.join(audiofiles,currentfile), methodfiles[0], methodfiles[1], trialtitle).show()
                # response = responsewindow(root, os.path.join(audiofiles,currentfile), trialtitle).show()
                if response:
                    flg=1

def run_trials(**kwargs):
    nt = int(kwargs.get('ntrials').get())
    audiofiles = kwargs.get('audiofiles').get()
    ID = kwargs.get('id').get()
    root = kwargs.get('rootfig')
    grate= float(kwargs.get('grate').get())
    mrate = float(kwargs.get('mrate').get())
    outdir = kwargs.get('outdir').get()
    minsnr= float(kwargs.get('minsnr').get())
    maxsnr= float(kwargs.get('maxsnr').get())

    method = kwargs.get('method').get()
    methodfiles=kwargs.get('methodfiles').get().split(',')

    modelp=np.array([[0.5], [mrate], [grate], [minsnr], [maxsnr], [0], [0.5]])

    slopeweight=kwargs.get('slopeweight').get()
    plotarea=kwargs.get('plot')

    basiep={'cs': slopeweight}
    reverblist, filelist, availsnr = extractfilelist(audiofiles)

    if not filelist :
        print('No .wav files in provided folder')
    elif not availsnr:
        print('Wrong snr naming convention in provided folder')
    else:
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")#
        nmodels = len(reverblist)

        if nmodels>5:
            tmp=messagebox.askquestion("Warning!", f"The experiment contains {nmodels} models. This will have slow convergence. Continue?", icon="question")
            if tmp=='no':
                return

        if nmodels==1:
            snrlist=np.sort(list(set(availsnr[0])))
        else:
            snrlist=[]
            for i in range(nmodels):
                snrlist.append(np.sort(list(set(availsnr[i]))))
            snrlist = np.asarray(snrlist)
        # ------------ initialise model ------------
        IDmatches = check_ID_day(outdir, ID)
        if IDmatches:
            IDmatches.insert(0, 'New')
            selectedID = listselect(root, IDmatches).show()
            
            if selectedID=='New':
                srt_estimator=basie_estimator()
                [snr, evalmodel,_,_] = srt_estimator.initialise(nmodels, modelp=np.repeat(modelp, nmodels, axis=1), basiep=basiep, availsnr=snrlist)
                filenames=[]
                choices=[]
            else:
                if os.path.isfile(os.path.join(selectedID,'finished.pkl')):
                    pklfile=os.path.join(selectedID,'finished.pkl')
                elif os.path.isfile(os.path.join(selectedID,'paused.pkl')):
                    pklfile=os.path.join(selectedID,'paused.pkl')
                else:
                    print('no .pkl file')

                with open(pklfile, 'rb') as f:
                    [srt_estimator, evalmodel, snr, filenames, choices]=pickle.load(f)
        else:
            srt_estimator=basie_estimator()
            [snr, evalmodel,_,_] = srt_estimator.initialise(nmodels, modelp=np.repeat(modelp, nmodels, axis=1), basiep=basiep, availsnr=snrlist)
            filenames=[]
            choices=[]

        os.makedirs(os.path.join(outdir, ID, timestamp), exist_ok=True)
        varcount=0
        # ------------ plot parameters ------------
        plotsnr=[]
        plotarea.fig.clf()
        ax = plotarea.fig.add_subplot(111, xlabel='Trial number (per model)', ylabel='probe SNR [dB]', ylim=(modelp[3],modelp[4]))
        lines = []
        linessrt=[]
        # colors=['rx--','bx-.', 'gx:', ]
        linetypes=['--', '-.', ':']
        colors=['r','b','g','c', 'm', 'y']

        linecolors=[]
        colorlist=[]
        for i in range(nmodels):
            cflg=0
            while cflg==0:
                candc = random.choice(colors)
                cand = 'x' + random.choice(linetypes) + candc
                if candc not in linecolors:
                    linecolors.append(cand)
                    colorlist.append(candc)
                    cflg=1

        infostring=''
        for i in range(nmodels):
            # line,= ax.step(0,float('nan'),linecolors[i], label='Model: ' + str(i+1) + ' (' + reverblist[i] + ')', linewidth=1.6)
            # lineSRT = ax.axhline(y=0, color=colorlist[i], label='Model: ' + str(i+1) + ' (' + reverblist[i] + '), SRT', linewidth=1.6)
            line,= ax.step(0,float('nan'),linecolors[i], label='Model: ' + str(i+1) + ', probe', linewidth=1.6)
            lineSRT = ax.axhline(y=0, color=colorlist[i], label='Model: ' + str(i+1) + ', SRT', linewidth=1.6)
            lines.append(line)
            linessrt.append(lineSRT)
            infostring+=f"Model {str(i+1)}: SRT 00.00 \u00B1 00.00 dB\t Slope 00.00 \u00B1 00.00 dB\u207B\u00B9 \n" 
        ax.legend()
        # plotarea.fig.subplots_adjust(bottom=0.15, left=0.15)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid()
        infostring=infostring[0:-2]
        plotarea.textvar.set(infostring)
        if nmodels>1:
            ax.set_xlim([0, int(0.5*nt)+5])
        else:
            ax.set_xlim([0, nt])
        plotarea.activatepause()
        # ax.ylim([-20, 20])
        # -----------------------------------------

        for i in range(int(nt)):
            flg=0
            # plot update
            x,y = lines[evalmodel-1].get_data()
            lines[evalmodel-1].set_xdata(np.append(x,x[-1]+1))
            lines[evalmodel-1].set_ydata(np.append(y,snr))
            # lines[evalmodel-1], = ax.step(np.append(x,x[-1]+1), np.append(y,snr), linecolors[evalmodel-1], linewidth=1.6)
            plotarea.canvas.draw()

            # -- get response from response window --
            snridx = np.where(np.in1d(availsnr[evalmodel-1], snr))[0];
            try: # try unique options
                templist=[]
                for j in snridx:
                    if filelist[evalmodel-1][j] not in filenames:
                        templist.append(filelist[evalmodel-1][j])
                currentfile=random.choice(templist)
            except: # if you have tried everything already
                currentfile=random.choice([filelist[evalmodel-1][j] for j in snridx])


            filenames.append(currentfile)
            trialtitle = 'Trial ' + str(i+1) +'/' + str(nt)
            plotarea.filevar.set("Playing: "+currentfile)
            while flg==0:
                if method=='MRT [Hurr.]':
                    response, wordchoice = responsewindow_hurricane(root, os.path.join(audiofiles,currentfile), methodfiles[0], methodfiles[1], trialtitle).show()

                if plotarea.pausevar.get()=='paused':
                    flg=2
                elif response:
                    flg=1

            choices.append(wordchoice)
            if plotarea.pausevar.get()=='paused':
                print('experiment paused')
                with open(os.path.join(outdir, ID, timestamp, 'paused.pkl'),'wb') as f:
                    pickle.dump([srt_estimator, evalmodel, snr, filenames[0:-1], choices[0:-1]],f)
                break
            print('snr', snr, 'response', np.array([[(response=='1')]]), 'model', evalmodel)

            preevalmodel = evalmodel
            # calculate next snr and model
            [snr, evalmodel, m, v] = srt_estimator.update(evalmodel-1, probesnr=snr, response=int(response=='1'))

            # update plot area
            infostring=''
            for j in range(nmodels):
                infostring+=f'Model {j+1}: SRT {m[0,j,0]:2.2f} \u00B1 {np.sqrt(v[0,j]):2.2f} dB\t Slope {m[1,j,0]:2.2f} \u00B1 {np.sqrt(v[2,j]):2.2f} dB\u207B\u00B9 \n'
            infostring=infostring[0:-2]
            plotarea.textvar.set(infostring)

            linessrt[preevalmodel-1].set_ydata([m[0,preevalmodel-1, 0], m[0,preevalmodel-1, 0]])
            plotarea.canvas.draw()

            if all(v[0,:]<1):
                varcount+=varcount
                if varcount>4:
                    print('var is low enough')
                    break
            else:
                varcount=0
        [p, q, msr] = srt_estimator.summary()

        ####### save a config file with details of experiment #######
        with open(os.path.join(outdir, ID, timestamp, 'config.txt'), 'w') as file:
            file.write(f'time: {timestamp} \t subjectID: {ID} \t audio folder: {audiofiles} \t model parameters: {modelp.flatten()} \t reverb list: {reverblist} \t ntrials: {nt}')

        ####### save results in a csv file #######
        nlines = len(msr[:,0])
        print(nlines)
        print(msr)
        print(filenames)
        df=pd.DataFrame({'ID': [ID] * nlines, 'time': [timestamp]*nlines, 'model nbr': msr[:,0], 'snr (dB)': msr[:,1], 'file':filenames[0:nlines], 'choice':choices[0:nlines],
         'correct?': msr[:,2],'srt': msr[:,3], 'log-slope': msr[:,4], 'var(srt)': msr[:,5], 'var(log-slope)': msr[:,6]})

        df.to_csv(os.path.join(outdir, ID, timestamp, 'results.csv'), index=False)

        if plotarea.pausevar.get()=='active':
            with open(os.path.join(outdir, ID, timestamp, 'finished.pkl'),'wb') as f:
                pickle.dump([srt_estimator, evalmodel, snr, filenames, choices],f)

        plotarea.hidepause()
        plotarea.pausevar.set('active')

class responsewindow_hurricane(Toplevel):
    def __init__(self, root, audiofile, reftxt, senttxt,  titlestr):
        # need to add response buttons based on info for list in .txt files
        # the audiofile should define the true/false value of buttons
        # start panel should have a rolldown menu giving the type of feedback needed
        Toplevel.__init__(self, root)
        self.title(titlestr)
        self.geometry("+%d+%d" %(root.winfo_x()+400, root.winfo_y()+300))

        self.responseVar = StringVar(value='')

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.grid_columnconfigure(0, weight=1)

        self.audioVar=StringVar()
        self.audiobtn = launchbutton(self, play_audio, 'Play', [0,0,1,1], audiofile=audiofile, audioVar=self.audioVar)

        shortaudio=audiofile.split("/")[-1]
        if "practice" not in shortaudio:
            fileID="_".join(shortaudio.split("_", 2)[:2])
        else:
            fileID="_".join(shortaudio.split("_", 3)[1:3])
        print(fileID)
        fileID.replace('\n', '')
        with open(reftxt) as file:
            for num, line in enumerate(file, 1):
                if fileID in line:
                    self.listnbr=num-1
                    print(line.split('\t'))
                    for i in range(len(line.split('\t')[1:])):
                        if fileID in line.split('\t')[i+1]:
                            idxinlist=i

        with open(senttxt) as file:
            self.sentences=file.readlines()[self.listnbr].split('\t')[1:]

        self.correctsentence=self.sentences[idxinlist]
        print(fileID, self.correctsentence)
        self.responsebtns = responsebutton_wordselect(self, self.responseVar, self.sentences, [1,0,1,1])

        # self.confirmframe = confirmbutton(self.parent, confirmVar, [2,0,1,1])
        self.confirmbutton = Button(self, text='Confirm', width=10, command=lambda: self.confirmresponse())

        self.confirmbutton.grid(row=2, column=0)

    def confirmresponse(self):
        if not self.audioVar.get():
            print('listen to the audio!')
        elif not self.responseVar.get():
            print('pick a response!')
        else:
            self.destroy()

    def show(self):
        self.wm_deiconify()
        # self.responsebtns.focus_force()
        self.wait_window()
        if not self.responseVar.get():
            return '', self.responseVar.get()
        elif self.responseVar.get()==self.correctsentence:
            return '1', self.responseVar.get()
        else:
            return '0', self.responseVar.get()

    def closewindow(self):
        self.destroy()

class responsewindow(Toplevel):
    def __init__(self, root, audiofile, titlestr):
        
        Toplevel.__init__(self, root)
        self.title(titlestr)
        self.geometry("+%d+%d" %(root.winfo_x()+500, root.winfo_y()+300))

        self.responseVar = StringVar(value='')

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.grid_columnconfigure(0, weight=1)

        self.audioVar=StringVar()
        self.audiobtn = launchbutton(self, play_audio, 'Play', [0,0,1,1], audiofile=audiofile, audioVar=self.audioVar)

        self.responsebtns = responsebutton(self, self.responseVar, [1,0,1,1])

        # self.confirmframe = confirmbutton(self.parent, confirmVar, [2,0,1,1])
        self.confirmbutton = Button(self, text='Confirm', width=10, command=lambda: self.confirmresponse())

        self.confirmbutton.grid(row=2, column=0)

    def confirmresponse(self):
        if not self.audioVar.get():
            print('listen to the audio!')
        elif not self.responseVar.get():
            print('pick a response!')
        else:
            self.destroy()

    def show(self):
        self.wm_deiconify()
        # self.responsebtns.focus_force()
        self.wait_window()
        return self.responseVar.get()

    def closewindow(self):
        self.destroy()
class responsebutton:
    def __init__(self, root, vaript=None, pos=[0,0,1,1]):
        self.parent = Frame(root) # frame to hold the box and labels

        if vaript is None:
            self.var = StringVar()
        else :
            self.var = vaript # value of the entry

        self.boxyes = Radiobutton(self.parent, text='Yes', width=10, variable=vaript, value=1)
        self.boxno = Radiobutton(self.parent, text='No', width=10, variable=vaript, value=0)

        self.place_on_grid(pos) # place things on the grid


    def place_on_grid(self, newpos):
        self.parent.grid(row=newpos[0], column=newpos[1], rowspan=newpos[2], columnspan=newpos[3], sticky='n', padx=10,pady=10)
        self.parent.grid_rowconfigure(0,weight=1)
        self.parent.grid_columnconfigure(0,weight=1)

        self.boxyes.grid(row=0, column=0, sticky='s')
        self.boxno.grid(row=0, column=1, sticky='s')

    def update_val(self, newval):
        self.var.set(newval)
class responsebutton_wordselect:
    def __init__(self, root, vaript=None, sentences='', pos=[0,0,1,1]):
        self.parent = Frame(root) # frame to hold the box and labels

        if vaript is None:
            self.var = StringVar()
        else :
            self.var = vaript # value of the entry

        self.buttons=[]
        for i in range(len(sentences)):
            self.buttons.append(Radiobutton(self.parent, text=sentences[i].replace('\n', ''), variable=vaript, value=sentences[i].replace('\n', ''), justify='left', font=('Arial', 20)))
        self.inaudible=Radiobutton(self.parent, text="Don't know", variable=vaript, value="Don't know", font=('Arial, 20'))
        self.place_on_grid(pos) # place things on the grid

    def place_on_grid(self, newpos):
        self.parent.grid(row=newpos[0], column=newpos[1], rowspan=newpos[2], columnspan=newpos[3], sticky='n', padx=10,pady=10)
        self.parent.grid_rowconfigure(0,weight=1)
        self.parent.grid_rowconfigure(1,weight=1)
        self.parent.grid_columnconfigure(0,weight=1)

        for i in range(len(self.buttons)):
            self.buttons[i].grid(row=0, column=i, sticky='ws', padx=30)
        
        self.inaudible.grid(row=0, column=i+1, sticky='s')
    def update_val(self, newval):
        self.var.set(newval)

class browsebutton:
    def __init__(self, root, lbl='Default', vaript=None, pos=[0,0,1,1]):
        self.parent = Frame(root) # frame to hold the box and labels

        if vaript is None:
            self.var = StringVar()
        else :
            self.var = vaript # value of the entry
        self.__lblfull = StringVar(self.parent, value='Value: ' +str(self.var.get()))


        self.title = Label(self.parent, text=lbl, anchor='n', font=('Arial', 12, 'bold')) # title
        # self.box = Edit(self.parent, textvariable=self.var, justify='right', width=12, validate='key', validatecommand=vcmd, invalidcommand=ivcmd) # box entry
        self.box = Button(self.parent, text='Browse...', command=self.browse_computer, width=10)
        self.varlbl = Message(self.parent, font=('Arial', 12), textvariable=self.__lblfull) # label with var value

        self.place_on_grid(pos) # place things on the grid

        self.var.trace('w', self.__update_varlbl)

    def browse_computer(self):
        # self.parent.withdraw()
        folder_selected = filedialog.askdirectory(initialdir=os.getcwd())
        self.var.set(folder_selected)
    

    def place_on_grid(self, newpos):
        self.parent.grid(row=newpos[0], column=newpos[1], rowspan=newpos[2], columnspan=newpos[3], sticky='n', padx=10,pady=10)
        self.parent.grid_rowconfigure(0,weight=1)

        self.title.grid(row=0, column=0, sticky='w')
        self.box.grid(row=1, column=0, sticky='s')
        self.varlbl.grid(row=2,column=0, sticky='nw')

    def update_val(self, newval):
        self.var.set(newval)

    def __update_varlbl(self, *args):
        self.__lblfull.set('Value: ' +str(self.var.get()))
class browsebuttonfile:
    def __init__(self, root, lbl='Default', vaript=None, pos=[0,0,1,1]):
        self.parent = Frame(root) # frame to hold the box and labels

        if vaript is None:
            self.var = StringVar()
        else :
            self.var = vaript # value of the entry
        self.__lblfull = StringVar(self.parent, value='Value: ' +str(self.var.get()))


        self.title = Label(self.parent, text=lbl, anchor='n', font=('Arial', 12, 'bold')) # title
        # self.box = Edit(self.parent, textvariable=self.var, justify='right', width=12, validate='key', validatecommand=vcmd, invalidcommand=ivcmd) # box entry
        self.box = Button(self.parent, text='Browse...', command=self.browse_computer, width=10)
        self.varlbl = Message(self.parent, font=('Arial', 12), textvariable=self.__lblfull) # label with var value

        self.place_on_grid(pos) # place things on the grid

        self.var.trace('w', self.__update_varlbl)

    def browse_computer(self):
        # self.parent.withdraw()
        folder_selected = filedialog.askopenfilename(initialdir=os.getcwd())
        self.var.set(folder_selected)
    

    def place_on_grid(self, newpos):
        self.parent.grid(row=newpos[0], column=newpos[1], rowspan=newpos[2], columnspan=newpos[3], sticky='n', padx=10,pady=10)
        self.parent.grid_rowconfigure(0,weight=1)

        self.title.grid(row=0, column=0, sticky='w')
        self.box.grid(row=1, column=0, sticky='s')
        self.varlbl.grid(row=2,column=0, sticky='nw')

    def __update_varlbl(self, *args):
        self.__lblfull.set('Value: ' +str(self.var.get()))
class launchbutton:
    def __init__(self, root, launchfct=defaultfct,lbl='Default',pos=[0,0,1,1], **kargs):
        self.parent = Frame(root) # frame to hold the box and labels
        self.launchfct = launchfct

        self.box = Button(self.parent, text=lbl, command=lambda: self.start_function(launchfct,**kargs), width=7, font=(17))

        self.place_on_grid(pos) # place things on the grid

    def place_on_grid(self, newpos):
        self.parent.grid(row=newpos[0], column=newpos[1], rowspan=newpos[2], columnspan=newpos[3], sticky='n', padx=10,pady=10)
        self.parent.grid_rowconfigure(0,weight=1)

        self.box.grid(row=0, column=0, sticky='s')

    def start_function(self,launchfct, **kwargs):
        self.box['state']='disabled'
        launchfct(**kwargs)
        self.box['state']='normal'
class plotarea:
    def __init__(self, root, **kargs):
        self.fig=plt.Figure()
        # self.ax=self.fig.add_subplot()
        # self.line, = self.ax.plot(0,0)

        # current file info
        self.filevar=StringVar(value="Playing:")
        self.currentfile=Label(root, textvariable=self.filevar, font=('Arial', 13))
        self.currentfile.grid(row=0, column=0, sticky='w')

        # plot
        self.canvas=FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky='w')

        # estimate values
        self.textvar=StringVar(value='')
        self.__label=Message(root, textvariable=self.textvar, font=('Arial',13, 'bold'), width=600)
        self.__label.grid(row=2,column=0, sticky='we')

        # pause buttons
        self.pausevar=StringVar(value='active', master=root)
        self.pausebtn=launchbutton(root, self.pauseexperiment, 'Pause', [3,0,1,1])
        self.hidepause
        # self.pausebtn.grid(row=2,column=0, sticky='we')

    def activatepause(self):
        self.pausebtn.place_on_grid([3,0,1,1])

    def hidepause(self):
        self.pausebtn.box.grid_forget()

    def pauseexperiment(self):
        self.pausevar.set('paused')
        windowitems=[]
        for k,v in self.canvas.get_tk_widget().master.master.children.items():
            if 'responsewindow' in k:
                windowitems.append(v)
        for i in windowitems:
            i.closewindow()
class textentry:
    def __init__(self, root, lbl='Default', vaript=None, pos=[0,0,1,1], range=None):
        self.parent = Frame(root) # frame to hold the box and labels

        if vaript is None:
            self.var = StringVar()
        else :
            self.var = vaript # value of the entry
        self.__lblfull = StringVar(self.parent, value='Value: ' +str(self.var.get()))
        self.__datatype=self.__check_datatype()

        self.__range = range

        vcmd = (self.parent.register(self.__callback), '%P')
        ivcmd = (self.parent.register(self.__on_invalid), '%P')

        self.title = Label(self.parent, text=lbl, anchor='n', font=('Arial', 12, 'bold')) # title
        self.box = Entry(self.parent, textvariable=self.var, justify='right', width=12, validate='key', validatecommand=vcmd, invalidcommand=ivcmd) # box entry
        self.varlbl = Label(self.parent, textvariable=self.__lblfull, font=('Arial', 12)) # label with var value

        self.place_on_grid(pos) # place things on the grid

        self.var.trace('w', self.__update_varlbl)

    def __callback(self, P):
        self.title.config(fg='black')
        self.title['text']= self.title['text'].strip('('+self.__datatype+')')
        if self.__range is not None:
            self.title['text'] = self.title['text'].strip(' in ('+str(self.__range[0])+','+str(self.__range[1])+')')
        if P=="" or self.__datatype =='string':
            return True
        else:
            try:
                prompt = self.__datatype +'(P)'
                value=eval(prompt)
                if self.__range is None:
                    return True
                elif value>=self.__range[0] and value<=self.__range[1]:
                    return True
                else:
                    return False
            except:
                return False

    def __on_invalid(self, P):
        self.title.config(fg='red')
        try:
            prompt = self.__datatype +'(P)'
            eval(prompt)
            self.title['text'] += ' in ('+str(self.__range[0])+','+str(self.__range[1])+')'
        except:
            self.title['text'] += '('+self.__datatype+')'

    def __check_datatype(self):
        try:
            int(self.var.get())
            return 'int'
        except ValueError:
            try:
                float(self.var.get())
                return 'float'
            except ValueError:
                return 'string'

    def place_on_grid(self, newpos):
        self.parent.grid(row=newpos[0], column=newpos[1], rowspan=newpos[2], columnspan=newpos[3], sticky='n', padx=10,pady=10)
        self.parent.grid_rowconfigure(0,weight=1)

        self.title.grid(row=0, column=0, sticky='w')
        self.box.grid(row=1, column=0, sticky='s')
        self.varlbl.grid(row=2,column=0, sticky='nw')

    def __update_varlbl(self, *args):
        self.__lblfull.set('Value: ' +str(self.var.get()))
class listselect(Toplevel):    
    def __init__(self, root, filelist):
        
        Toplevel.__init__(self, root)
        self.title('Select folder')
        self.geometry("+%d+%d" %(root.winfo_x()+500, root.winfo_y()+300))

        self.responseVar = StringVar(value='New')

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.label = Label(self, text='An experiment already exists for this ID today - please select what to load', font=('Arial', 12))
        self.label.grid(row=0,column=0)
        self.listbox = ttk.Combobox(self, values=filelist, textvariable=self.responseVar)
        self.listbox.grid(row=1, column=0)

        # self.responsebtns = responsebutton(self, self.responseVar, [1,0,1,1])

        # self.confirmframe = confirmbutton(self.parent, confirmVar, [2,0,1,1])
        self.confirmbutton = Button(self, text='Confirm', width=10, command=lambda: self.confirmresponse())

        self.confirmbutton.grid(row=2, column=0)

    def confirmresponse(self):
        self.destroy()

    def show(self):
        self.wm_deiconify()
        # self.responsebtns.focus_force()
        self.wait_window()
        return self.responseVar.get()
class dropdownmenu():
    def __init__(self, root, lbl='Default', vaript=None, menuopts=['Default'], pos=[0,0,1,1]):
        self.parent = Frame(root)

        if vaript is None:
            self.var = StringVar()
        else :
            self.var = vaript # value of the entry
        

        self.methodfiles=[]
        self.methodfilesvar=StringVar(self.parent)
        self.title = Label(self.parent, text=lbl, anchor='n', font=('Arial', 12, 'bold')) # title
        self.menu = OptionMenu(self.parent, self.var, *menuopts, command=self.getfiles)
        self.varlbl = Label(self.parent, textvariable=StringVar(value=' '), font=('Arial', 12))
        self.menu.config(width=8, height=1)
        self.place_on_grid(pos)

        if self.var.get()=='MRT [Hurr.]':
            idfilevar = StringVar(self.parent,value='recordings.txt')
            sentencefilevar = StringVar(self.parent,value='sentences.txt')
            self.methodfiles.append(browsebuttonfile(self.parent.master, 'List ID: ', idfilevar, [1,1,1,1]))
            self.methodfiles.append(browsebuttonfile(self.parent.master, 'List sentences: ', sentencefilevar, [1,2,1,1]))
            self.methodfilesvar.set(idfilevar.get()+ ","+ sentencefilevar.get())

    def place_on_grid(self, newpos):
        self.parent.grid(row=newpos[0], column=newpos[1], rowspan=newpos[2], columnspan=newpos[3], sticky='n', padx=10,pady=10)
        self.parent.grid_rowconfigure(0,weight=1)
        self.parent.grid_columnconfigure(0,weight=1)

        self.title.grid(row=0, column=0, sticky='w')
        self.menu.grid(row=1, column=0, sticky='s', pady=4)
        self.varlbl.grid(row=2,column=0, sticky='nw')

    def getfiles(self, event):
        # print(self.var.get())
        for i in self.methodfiles:
            i.parent.destroy()

        self.methodfiles=[]
        # self.methodfilesvar=[]
        if self.var.get()=='MRT [Hurr.]':
            idfilevar = StringVar(self.parent,value='recordings.txt')
            sentencefilevar = StringVar(self.parent,value='sentences.txt')
            self.methodfiles.append(browsebuttonfile(self.parent.master, 'List ID: ', idfilevar, [1,1,1,1]))
            self.methodfiles.append(browsebuttonfile(self.parent.master, 'List sentences: ', sentencefilevar, [1,2,1,1]))
            self.methodfilesvar.set(idfilevar.get()+ ","+ sentencefilevar.get())
        else:
            print('other')

if __name__=='__main__':
    # --------- instantiate GUI ---------
    mainfig = Tk()
    mainfig.title("Psychometric function evaluation")

    # --------- create all main frames ---------
    imgframe=Frame(mainfig, width=600, height=150)
    paramframe=LabelFrame(mainfig, width=600, height=300, text="Parameters", font=('Arial', 14, 'bold'), labelanchor='n')
    advancedparamframe=LabelFrame(mainfig, width=600, height=150, text="Advanced parameters", font=('Arial', 14, 'bold'), labelanchor='n')   
    experframe=LabelFrame(mainfig, width=600, height=100, labelanchor='n')
    plotframe=Frame(mainfig, width=200, height=100)
    # pauseframe=Frame(mainfig, width=200, height=100)

    mainfig.grid_rowconfigure(0, weight=1)
    mainfig.grid_rowconfigure(1, weight=1)
    mainfig.grid_rowconfigure(2, weight=1)
    mainfig.grid_rowconfigure(3, weight=1)
    mainfig.grid_columnconfigure(0, weight=1)
    mainfig.grid_columnconfigure(1, weight=1)

    imgframe.grid(row=0, column=0, columnspan=2, padx=(20,30), pady=0, sticky='nsew')
    paramframe.grid(row=1, column=0, padx=20, pady=0, sticky='nsew')
    advancedparamframe.grid(row=2, column=0, padx=20, pady=0, sticky='nsew')
    experframe.grid(row=3, column=0, padx=20, pady=20, sticky='nsew')
    plotframe.grid(row=1, column=1, rowspan=3, padx=20, pady=10, sticky='nsew')
    # pauseframe.grid(row=3, column=1, padx=20, pady=0, sticky='nsew')

    # --------- add logos ---------
    imgframe.grid_columnconfigure(0, weight=1)
    imgframe.grid_rowconfigure(0, weight=1)

    iclimg=Image.open("icllogo.png")
    iclimg.thumbnail((168,150), Image.Resampling.LANCZOS)
    iclrender = ImageTk.PhotoImage(iclimg, master=imgframe)
    icllogo=Label(imgframe, image=iclrender).grid(row=0, column=0, sticky='w')

    uclimg=Image.open("ucllogo.png")
    uclimg.thumbnail((150,150), Image.Resampling.LANCZOS)
    uclrender = ImageTk.PhotoImage(uclimg, master=imgframe)
    ucllogo=Label(imgframe, image=uclrender).grid(row=0, column=1, sticky='e')

    # --------- Parameters ---------
    paramframe.grid_rowconfigure(0, weight=1)
    paramframe.grid_rowconfigure(1, weight=1)
    # paramframe.grid_rowconfigure(2, weight=1)

    paramframe.grid_columnconfigure(0, weight=1)
    paramframe.grid_columnconfigure(1, weight=1)
    paramframe.grid_columnconfigure(2, weight=1)


    audiofilevar=StringVar(paramframe, value='/Users/emiliedolne/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD/Year 3/Smartter hear/psychometrics/audio/mrt')
    audiobtn=browsebutton(paramframe, 'Audio files: ', audiofilevar, [0,1,1,1])

    outputdirvar=StringVar(paramframe, value='results')
    outputdir=browsebutton(paramframe, 'Output directory: ', outputdirvar, [0,2,1,1])

    subjectIDvar=StringVar(paramframe, value='ID')
    subjectID=textentry(paramframe, 'Subject ID: ', subjectIDvar, [0,0,1,1])

    methodvar=StringVar(paramframe, value='MRT [Hurr.]')
    methodmenu=dropdownmenu(paramframe, 'Test type: ', methodvar, ['MRT [Hurr.]', 'other'], [1,0,1,1])
    # methodvar.set('MRT [Hurr.]')

    # reffilesvar=StringVar(paramframe, value=['reference.txt', 'sentences.txt'])
    # reffiles=browsebuttonfiles(paramframe, 'Reference files: ', reffilesvar, [1,1,1,1])

    # --------- Advanced parameters ---------
    advancedparamframe.grid_rowconfigure(0, weight=1)
    advancedparamframe.grid_columnconfigure(0, weight=1)
    advancedparamframe.grid_columnconfigure(1, weight=1)
    advancedparamframe.grid_columnconfigure(2, weight=1)

    slopeweightvar=StringVar(advancedparamframe, value='0.5')
    slopeweight=textentry(advancedparamframe, 'Slope weight: ', slopeweightvar, [0,0,1,1], [0, 1])

    minsnrvar=StringVar(advancedparamframe, value='-20.0')
    minsnr=textentry(advancedparamframe, 'Min SNR (dB): ', minsnrvar, [0,1,1,1])

    maxsnrvar=StringVar(advancedparamframe, value='20.0')
    maxsnr=textentry(advancedparamframe, 'Max SNR (dB): ', maxsnrvar, [0,2,1,1], [0, 1])

    ntrialsvar=StringVar(advancedparamframe, value=4)
    ntrials=textentry(advancedparamframe, 'Max nbr. trials: ', ntrialsvar, [1,2,1,1])

    guessratevar=StringVar(advancedparamframe, value=0.1)
    guessrate=textentry(advancedparamframe, 'Guess rate: ', guessratevar, [1,0,1,1], [0, 1])

    missratevar=StringVar(advancedparamframe, value=0.04)
    missrate=textentry(advancedparamframe, 'Miss rate: ', missratevar, [1,1,1,1], [0, 1])

    ntrialspractice=4
    # --------- Control area ---------
    plotframe.grid_rowconfigure(0, weight=1)
    plotframe.grid_rowconfigure(1, weight=1)
    plotframe.grid_rowconfigure(2, weight=1)
    plotframe.grid_columnconfigure(0, weight=1)
    canvas=plotarea(plotframe)
    canvas.hidepause()

    # --------- Run buttons ---------
    experframe.grid_rowconfigure(0, weight=1)
    experframe.grid_columnconfigure(0, weight=1)
    experframe.grid_columnconfigure(1, weight=1)

    practicebtn=launchbutton(experframe, run_practice, 'Practice', [0,0,1,1], ntrials=ntrialspractice, audiofiles=audiofilevar, rootfig=mainfig, method=methodvar, 
        methodfiles=methodmenu.methodfilesvar)

    runbutton=launchbutton(experframe, run_trials, 'Start', [0,1,1,1], id=subjectIDvar, ntrials=ntrialsvar, audiofiles=audiofilevar, 
        rootfig=mainfig, mrate=missratevar, grate=guessratevar, outdir=outputdirvar, slopeweight=slopeweightvar, plot=canvas, minsnr=minsnrvar, maxsnr=maxsnrvar,
        method=methodvar, methodfiles=methodmenu.methodfilesvar)



    mainfig.mainloop()


