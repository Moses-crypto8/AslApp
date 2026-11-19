"""Visible in-camera controls; mouse clicks do not rely on keyboard focus."""
import cv2

class CameraControls:
    def __init__(self,window):
        self.window=window;self.pending=None;self.boxes={};self.enabled={};self.bound=False

    def on_mouse(self,event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONUP:return
        for action,(x1,y1,x2,y2) in self.boxes.items():
            if self.enabled.get(action,False) and x1<=x<=x2 and y1<=y<=y2:
                self.pending=action
                return

    def show(self,image,capture):
        h,w=image.shape[:2]
        canvas=cv2.copyMakeBorder(image,0,100,0,max(0,640-w),cv2.BORDER_CONSTANT,value=(25,25,25))
        if capture.active:
            state='PAUSED - finish or discard' if capture.interrupted else 'RECORDING - click Finish sign'
            color=(30,50,220)
        elif capture.armed:
            state='WAITING FOR HANDS - raise hands to begin';color=(0,190,240)
        else:
            state='READY - camera preview only';color=(90,210,90)
        cv2.putText(canvas,state,(16,h+24),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
        self.boxes={'start':(12,h+38,170,h+88),'finish':(182,h+38,350,h+88),
                    'discard':(362,h+38,490,h+88),'quit':(502,h+38,630,h+88)}
        self.enabled={'start':not(capture.active or capture.armed),'finish':capture.active or capture.armed,
                      'discard':capture.active or capture.armed,'quit':True}
        for action,label in [('start','Start sign'),('finish','Cancel start' if capture.armed else 'Finish sign'),('discard','Discard'),('quit','Close')]:
            x1,y1,x2,y2=self.boxes[action]
            enabled=self.enabled[action]
            background=(55,100,190) if action=='finish' and enabled else ((65,65,65) if enabled else (35,35,35))
            cv2.rectangle(canvas,(x1,y1),(x2,y2),background,-1)
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(170,170,170) if enabled else (70,70,70),1)
            cv2.putText(canvas,label,(x1+10,y1+32),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255) if enabled else (105,105,105),1)
        cv2.imshow(self.window,canvas)
        if not self.bound:
            try:
                cv2.setMouseCallback(self.window,self.on_mouse)
                self.bound=True
            except cv2.error:
                # Allows non-GUI tests to mock imshow; a real window registers
                # its callback here immediately after the first presentation.
                pass

    def command(self,key):
        if self.pending is not None:
            action,self.pending=self.pending,None
            return action
        key=key&0xff
        if key in (32,13,10):return 'toggle'
        if key==27:return 'discard'
        if key==ord('q'):return 'quit'
        return None

    @staticmethod
    def should_toggle(command,capture):
        # Explicit mouse actions cannot accidentally restart a finished recording.
        return (command=='toggle' or
                (command=='start' and not(capture.active or capture.armed)) or
                (command=='finish' and (capture.active or capture.armed)))
