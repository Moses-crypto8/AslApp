"""Timed isolated-sign capture and shared resampling for collection/inference."""
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class Segment:
    frames: list
    times: list
    reason: str
    @property
    def duration(self):
        return self.times[-1]-self.times[0] if len(self.times)>1 else 0.0


def resample_segment(frames, times, count=15):
    """Map the COMPLETE recording to model frames using elapsed time.

    Interpolate only when a hand exists on both neighboring observations;
    otherwise choose the nearest observation for that hand, preserving absence.
    """
    a=np.asarray(frames,dtype=np.float64)
    t=np.asarray(times,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=84 or len(a)<2 or t.shape!=(len(a),):
        raise ValueError('A recording needs at least two timestamped 84-value frames.')
    if not np.isfinite(a).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Frames must be finite and timestamps strictly increasing.')
    if count<2:raise ValueError('Resampling requires at least two model frames.')
    t=t-t[0]
    target=np.linspace(0,t[-1],count)
    hi=np.clip(np.searchsorted(t,target,side='right'),1,len(t)-1);lo=hi-1
    weight=(target-t[lo])/(t[hi]-t[lo])
    hands=a.reshape(-1,2,42);out=np.zeros((count,2,42))
    for h in range(2):
        left,right=hands[lo,h],hands[hi,h]
        both=np.any(left!=0,axis=1)&np.any(right!=0,axis=1)
        out[:,h]=np.where((weight<=.5)[:,None],left,right)
        out[both,h]=left[both]*(1-weight[both,None])+right[both]*weight[both,None]
    return out.reshape(count,84)


def motion_rate(previous,current,dt):
    """Median landmark speed, in palm lengths per second; no camera-pixel threshold."""
    if previous is None or dt<=0:return 0.0
    a=np.asarray(previous).reshape(2,21,2);b=np.asarray(current).reshape(2,21,2)
    common=np.any(a!=0,axis=(1,2))&np.any(b!=0,axis=(1,2))
    if not common.any():return 0.0
    palms=np.linalg.norm(b[common,9]-b[common,0],axis=1)
    scale=float(np.median(palms))
    if scale<1e-8:return 0.0
    return float(np.max(np.median(np.linalg.norm(b[common]-a[common],axis=2),axis=1))/scale/dt)


class SignCapture:
    """Manual SPACE boundaries, or optional motion/pause heuristic.

    Automatic mode is for separated isolated signs, not linguistic sentence
    segmentation. Automatic mode discards long gaps; manual mode freezes the buffer so the user can finish or discard explicitly.
    """
    def __init__(self, mode='manual',max_seconds=6.,min_seconds=.35,
                 idle_seconds=.8,gap_seconds=.25,motion_threshold=1.2):
        if not np.isfinite([max_seconds,min_seconds,idle_seconds,gap_seconds,motion_threshold]).all():raise ValueError('Segmentation settings must be finite.')
        if mode not in ('manual','auto'):raise ValueError('Mode must be manual or auto.')
        if not 0<min_seconds<max_seconds<=30:raise ValueError('Require 0 < minimum < maximum duration <= 30 seconds.')
        if idle_seconds<=0 or gap_seconds<=0 or motion_threshold<=0:raise ValueError('Segmentation thresholds must be positive.')
        self.mode=mode;self.max_seconds=max_seconds;self.min_seconds=min_seconds
        self.idle_seconds=idle_seconds;self.gap_seconds=gap_seconds;self.motion_threshold=motion_threshold
        self.frames=[];self.times=[];self.active=False;self.status='SPACE: start signing'
        self.previous=None;self.previous_time=None;self.was_present=False
        self.missing_since=None;self.last_motion=0.;self.cooldown_until=-float('inf')
        self.pre=deque();self.clock=None;self.speed=0.;self.interrupted=False;self.armed=False

    def cancel(self,now,message='Recording discarded'):
        self.frames=[];self.times=[];self.active=False;self.pre.clear()
        self.missing_since=None;self.interrupted=False;self.armed=False;self.status=message;self.cooldown_until=now+.5

    def finish(self,now,reason):
        frames,times=self.frames,self.times
        if self.mode=='auto' and reason=='pause' and times:
            end=max(times[0]+self.min_seconds,self.last_motion+.15)
            keep=min(len(times),int(np.searchsorted(times,end,side="left"))+1)
            frames,times=frames[:keep],times[:keep]
        segment=Segment(frames,times,reason)
        self.cancel(now)
        if len(frames)<3 or segment.duration<self.min_seconds:
            self.status='Too short — record a complete sign, then SPACE to finish'
            return None
        self.status=f'Captured {segment.duration:.1f}s ({len(frames)} frames; {reason})'
        return segment

    def update(self,vector,now,toggle=False,cancel=False):
        now=float(now)
        if not np.isfinite(now) or (self.clock is not None and now<=self.clock):
            raise ValueError('Capture timestamps must be finite and strictly increasing.')
        self.clock=now
        if cancel:
            self.cancel(now);return None
        present=vector is not None and np.any(vector)
        if present:
            vector=np.asarray(vector,dtype=float)
            if vector.shape!=(84,) or not np.isfinite(vector).all():raise ValueError('Invalid landmark frame.')
        appeared=present and not self.was_present
        self.was_present=present
        dt=now-self.previous_time if self.previous_time is not None else 0
        rate=motion_rate(self.previous,vector,dt) if present else 0
        self.speed=.4*rate+.6*self.speed if present else 0.
        self.previous=vector.copy() if present else None;self.previous_time=now
        if not self.active:
            if self.armed and toggle:
                self.armed=False;self.status='Start cancelled — click Start sign when ready'
                return None
            if self.armed and not present:
                self.status='Waiting for hands — not recording yet; raise hands to begin'
                return None
            if present:
                self.pre.append((now,vector.copy()))
                while self.pre and now-self.pre[0][0]>.2:self.pre.popleft()
            else:self.pre.clear()
            start=toggle or (self.armed and present) or (self.mode=='auto' and now>=self.cooldown_until and present and (appeared or self.speed>=self.motion_threshold))
            if not start:return None
            if not present:
                self.armed=True
                self.status='Waiting for hands — not recording yet; raise hands to begin'
                return None
            self.active=True;self.armed=False;self.last_motion=now;self.missing_since=None;self.interrupted=False
            entries=list(self.pre) if self.mode=='auto' else [(now,vector.copy())]
            self.times=[t for t,_ in entries];self.frames=[v for _,v in entries]
            self.status='Recording — SPACE/Enter: finish | Esc: discard'
            return None
        # Stop has priority over tracking availability. Reaching for the keyboard
        # must not discard the good frames already recorded.
        long_gap = self.missing_since is not None and now-self.missing_since>=self.gap_seconds
        if toggle:
            if present and not self.interrupted and not long_gap:
                self.frames.append(vector.copy());self.times.append(now)
            return self.finish(now,'manual')
        if self.mode=='auto' and (long_gap or (not present and now-self.times[0]>=self.max_seconds)):
            self.cancel(now,'Tracking lost — recording discarded; try again')
            return None
        if now-self.times[0]>=self.max_seconds:
            return self.finish(now,'maximum duration')
        if not present:
            if self.missing_since is None:self.missing_since=now
            long_gap=now-self.missing_since>=self.gap_seconds
        if long_gap or self.interrupted:
            if self.mode=='auto':
                self.cancel(now,'Tracking lost — recording discarded; try again')
            else:
                # Do not join observations across a long gap. Freeze the existing
                # buffer until explicit finish/discard, even if hands return.
                self.interrupted=True
                self.status='Tracking paused — SPACE/Enter finish captured frames | Esc discard'
            return None
        if not present:
            self.status='Hands out of view — SPACE/Enter finishes captured frames'
            return None
        self.missing_since=None
        self.frames.append(vector.copy());self.times.append(now)
        if self.speed>=self.motion_threshold*.5:self.last_motion=now
        if self.mode=='auto' and now-self.last_motion>=self.idle_seconds:
            return self.finish(now,'pause')
        self.status=f'Recording {now-self.times[0]:.1f}/{self.max_seconds:.1f}s — SPACE/Enter: finish | Esc: discard'
        return None
