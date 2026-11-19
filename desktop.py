"""Small local desktop command center; camera windows run in a child process."""
from pathlib import Path
import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

ROOT = Path(__file__).resolve().parent

class Desktop:
    def __init__(self, root):
        self.root = root; self.process = None; self.events = queue.Queue()
        root.title('ASL Studio'); root.geometry('880x780'); root.minsize(850,740)
        style = ttk.Style(root); style.theme_use('clam')
        style.configure('TFrame', background='#f1f5f9')
        style.configure('TLabel', background='#f1f5f9', foreground='#0f172a', font=('Arial',11))
        style.configure('TButton', padding=8)
        frame = ttk.Frame(root, padding=22); frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='ASL Studio', font=('Arial',25,'bold')).pack(anchor='w')
        ttk.Label(frame,text='Recognize isolated signs • Collect samples • Retrain your model').pack(anchor='w',pady=(4,15))
        self.rating = tk.StringVar(); ttk.Label(frame,textvariable=self.rating,font=('Arial',13,'bold')).pack(anchor='w')
        self.refresh()
        ttk.Label(frame,text='Recorded-data holdout only; live and new-signer accuracy are unverified.',wraplength=760).pack(anchor='w',pady=(3,12))
        options = ttk.Frame(frame); options.pack(fill='x')
        self.camera = tk.StringVar(value='0'); self.threshold=tk.StringVar(value='0.55')
        self.width=tk.StringVar(); self.height=tk.StringVar(); self.engine=tk.StringVar(value='normalized')
        fields=[('Camera',self.camera),('Threshold',self.threshold),('Training width',self.width),('Training height',self.height)]
        for col,(name,var) in enumerate(fields):
            ttk.Label(options,text=name).grid(row=0,column=col,padx=4,sticky='w')
            ttk.Entry(options,textvariable=var,width=14).grid(row=1,column=col,padx=4,sticky='ew')
        ttk.Combobox(options,textvariable=self.engine,values=['normalized','gru'],state='readonly',width=12).grid(row=1,column=4,padx=6)
        ttk.Label(frame,text='Position/size normalization is automatic. GRU is the older optional experimental model.',wraplength=760).pack(anchor='w',pady=8)
        timing=ttk.Frame(frame);timing.pack(fill='x',pady=8)
        self.mode=tk.StringVar(value='manual');self.max_seconds=tk.StringVar(value='6.0')
        self.idle_seconds=tk.StringVar(value='0.8');self.motion_threshold=tk.StringVar(value='1.2')
        ttk.Label(timing,text='Boundaries').grid(row=0,column=0,sticky='w')
        ttk.Combobox(timing,textvariable=self.mode,values=['manual','auto'],state='readonly',width=12).grid(row=1,column=0,padx=(0,10))
        for col,(title,var) in enumerate([('Max seconds',self.max_seconds),('Auto pause (s)',self.idle_seconds),('Motion sensitivity',self.motion_threshold)],1):
            ttk.Label(timing,text=title).grid(row=0,column=col,sticky='w',padx=4)
            ttk.Entry(timing,textvariable=var,width=16).grid(row=1,column=col,padx=4)
        ttk.Label(frame,text='Use Start sign / Finish sign in the camera window. SPACE/Enter also works.',wraplength=800).pack(anchor='w')
        actions=ttk.Frame(frame); actions.pack(fill='x',pady=6)
        self.buttons=[]
        for title,action in [('Start recognition',self.recognize),('Train model',lambda:self.run(['train'])),('Train GRU',lambda:self.run(['train-temporal'])),('List terms',lambda:self.run(['terms'])),('Reproduce score',lambda:self.run(['evaluate']))]:
            button=ttk.Button(actions,text=title,command=action); button.pack(side='left',padx=(0,6)); self.buttons.append(button)
        capture=ttk.Frame(frame); capture.pack(fill='x',pady=12)
        self.label=tk.StringVar(); self.samples=tk.StringVar(value='30')
        ttk.Label(capture,text='New sign').grid(row=0,column=0,sticky='w')
        ttk.Entry(capture,textvariable=self.label,width=28).grid(row=1,column=0,padx=(0,8))
        ttk.Label(capture,text='Samples').grid(row=0,column=1,sticky='w')
        ttk.Entry(capture,textvariable=self.samples,width=7).grid(row=1,column=1,padx=(0,8))
        for col,(title,action) in enumerate([('Collect samples',self.collect),('Import CSV + retrain',self.update)],2):
            b=ttk.Button(capture,text=title,command=action);b.grid(row=1,column=col,padx=4);self.buttons.append(b)
        self.status=tk.StringVar(value='Ready. Model included; training is optional.')
        ttk.Label(frame,textvariable=self.status).pack(anchor='w',pady=(0,8))
        self.log=tk.Text(frame,height=15,wrap='word',bg='#0f172a',fg='#e2e8f0',font=('Courier',10),padx=12,pady=12)
        self.log.pack(fill='both',expand=True)
        ttk.Label(frame,text='Camera buttons: Start sign • Finish sign • Discard • Close. Ready means preview only.',wraplength=760).pack(anchor='w',pady=(8,0))
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(100,self.poll)

    def refresh(self):
        path=ROOT/'models/asl_model.json'
        try:
            m=json.loads(path.read_text())
            self.rating.set(f"Normalized validation: {m['accuracy']:.2%} | {m['terms']} terms | {m['rows']} samples")
        except (OSError,ValueError,KeyError): self.rating.set('Train a model to see validation accuracy')

    def camera_args(self):
        cam=int(self.camera.get())
        if cam < 0: raise ValueError('Camera must be nonnegative.')
        args=['--camera',str(cam)]
        w,h=self.width.get().strip(),self.height.get().strip()
        if w or h:
            if not w or not h or min(int(w),int(h)) <= 0: raise ValueError('Enter both positive training dimensions.')
            args += ['--width',w,'--height',h]
        maximum=float(self.max_seconds.get());idle=float(self.idle_seconds.get());motion=float(self.motion_threshold.get())
        if not .35<maximum<=30 or not 0<idle<=10 or not 0<motion<=100:
            raise ValueError('Maximum: > 0.35 and <= 30 s; pause: > 0 and <= 10 s; motion: > 0 and <= 100.')
        args += ['--mode',self.mode.get(),'--max-seconds',str(maximum),'--idle-seconds',str(idle),'--motion-threshold',str(motion)]
        return args

    def recognize(self):
        try:
            threshold=float(self.threshold.get())
            if not 0<=threshold<=1: raise ValueError('Threshold must be between 0 and 1.')
            self.run(['recognize','--engine',self.engine.get(),'--threshold',str(threshold)]+self.camera_args())
        except ValueError as exc: messagebox.showerror('Check settings',str(exc))

    def collect(self):
        try:
            if not self.label.get().strip(): raise ValueError('Enter a sign label.')
            n=int(self.samples.get())
            if n<1: raise ValueError('Samples must be positive.')
            cam=self.camera_args()
            out=filedialog.asksaveasfilename(initialdir=str(ROOT),defaultextension='.csv',filetypes=[('CSV','*.csv')])
            if out:self.run(['collect','--label',self.label.get().strip(),'--out',out,'--samples',str(n)]+cam)
        except ValueError as exc: messagebox.showerror('Check settings',str(exc))

    def update(self):
        path=filedialog.askopenfilename(initialdir=str(ROOT),filetypes=[('CSV','*.csv')])
        if path:self.run(['update-db','--update',path,'--retrain'])

    def run(self,args):
        if self.process is not None:return
        try:
            self.process=subprocess.Popen([sys.executable,'-u',str(ROOT/'app.py'),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        except OSError as exc:
            messagebox.showerror('Could not start',str(exc));return
        self.status.set('Running: '+args[0])
        for button in self.buttons:button.configure(state='disabled')
        proc=self.process
        def reader():
            for line in proc.stdout:self.events.put(('line',line))
            self.events.put(('done',proc.wait()))
        threading.Thread(target=reader,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=='line':self.log.insert('end',value);self.log.see('end')
                else:
                    self.process=None;self.refresh();self.status.set('Completed' if value==0 else 'Failed — see details below')
                    for b in self.buttons:b.configure(state='normal')
        except queue.Empty:pass
        self.root.after(100,self.poll)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()

def launch():
    root=tk.Tk();Desktop(root);root.mainloop()

if __name__=='__main__':launch()
