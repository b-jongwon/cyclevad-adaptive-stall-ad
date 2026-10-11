"""Bounded-history causal adapter for the frozen E11 C/P/Full model."""
from collections import deque
import numpy as np
from scipy.ndimage import convolve1d
from scipy.special import softmax
from .model import descriptor
from .paper_experiments import conditional_raw
from .tracker import circular_difference

class StreamingPhysicalTracker:
    def __init__(self, fitted, stride=2):
        if stride not in (1,2):raise ValueError('Supported source cadences are 1 and 2')
        self.fitted=fitted;self.stride=stride;g=fitted.grid_size
        offsets=np.arange(-g//4,g//4+1)
        sigma=np.sqrt(stride/2)*max(1,abs(g*2/fitted.period)*.5)
        self.weights=softmax(-.5*((offsets-g*stride/fitted.period)/sigma)**2)
        self.restart=1-(1-.001)**(stride/2)
        self.reset()
    def reset(self):
        g=self.fitted.grid_size;self.posterior=np.full(g,1/g);self.previous=np.full(g,1/g)
        self.x_history=deque(maxlen=4//self.stride+1)
        self.angle_history=deque(maxlen=32//self.stride+1)
        self.observation_history=deque(maxlen=32//self.stride+1)
        self.count=0
    def step(self,x,index):
        if index!=self.count*self.stride:raise ValueError('Out-of-order source index; reset before a new recording')
        t=self.fitted;g=t.grid_size;self.x_history.append(np.asarray(x).copy())
        encoded=np.r_[x,.5*(x-self.x_history[0])]
        distance=np.mean((encoded[None,:]-t.template)**2,axis=1)
        like=softmax(-distance/t.temperature);observed=(np.angle(like@t.unit)/(2*np.pi))%1
        innovation=one=0.
        if self.count:
            prior,other=convolve1d(np.stack([self.posterior,self.previous]),self.weights,axis=1,mode='wrap')
            prior=(1-self.restart)*prior+self.restart/g;other=(1-self.restart)*other+self.restart/g
            innovation=-np.log(max(g*np.dot(prior,like),1e-12));one=-np.log(max(g*np.dot(other,like),1e-12))
        else:prior=self.posterior
        self.posterior=prior*like;self.posterior/=max(self.posterior.sum(),1e-300)
        angle=(np.angle(self.posterior@t.unit)/(2*np.pi))%1;self.previous=like
        self.angle_history.append(angle);self.observation_history.append(observed)
        def progress(history):
            result=0.
            for h in [2,8,32]:
                k=h//self.stride
                if self.count<k:continue
                elapsed=h/t.period
                if elapsed<.45:result=max(result,abs(circular_difference(history[-1],history[-1-k])-elapsed)/max(elapsed,1/g))
            return float(result)
        out={'angle':float(angle),'observation_angle':float(observed),'alignment':float(distance.min()),'innovation':float(innovation),
             'one_innovation':float(one),'progress':progress(self.angle_history),'one_progress':progress(self.observation_history)}
        self.count+=1;return out

class StreamingDetector:
    def __init__(self,checkpoint,threshold,stride=2,readout='CP'):
        if readout not in ['CP','Full']:raise ValueError(readout)
        self.model=checkpoint;self.tracker=StreamingPhysicalTracker(checkpoint['base']['tracker'],stride)
        self.stride=stride;self.readout=readout;self.threshold=float(threshold);self.source_count=0;self.last=None
    def reset(self):self.tracker.reset();self.source_count=0;self.last=None
    def step(self,global_z=None,patches=None):
        index=self.source_count;self.source_count+=1
        if index%self.stride:
            if self.last is None:raise ValueError('No held prediction')
            return {**self.last,'source_index':index,'fresh':False}
        if global_z is None or patches is None:raise ValueError('Fresh observations require current features')
        m=self.model
        # Match the published cache's fp16 storage conversion, without reading a cache.
        global_z=np.asarray(global_z).astype(np.float16).astype(np.float32)
        patches=np.asarray(patches).astype(np.float16).astype(np.float32)
        z=descriptor({'global':global_z.reshape(1,-1),'patches':patches.reshape(1,*patches.shape)})
        x=(m['base']['tracker'].reducer.transform(z)/m['base']['tracker'].scale)[0]
        trace=self.tracker.step(x,index)
        co,ci,_=conditional_raw(z,np.array([trace['angle']]),m['coef'],m['h'],m['space'])
        c=max(m['ccal'][0].score(co)[0],m['ccal'][1].score(ci)[0])
        p=max(m['pcal'][k].score([trace[k]])[0] for k in ['alignment','innovation','progress'])
        a=None
        if self.readout=='Full':
            ao,ai,_=m['base']['pooled'].score(z);al,_=m['base']['local'].score(patches[None])
            a=max(m['appcal'][k].score(v)[0] for k,v in [('pooled',ao),('inside',ai),('local',al)])
        score=max(c,p) if a is None else max(a,c,p)
        threshold=self.threshold
        self.last={'source_index':index,'fresh':True,'theta':trace['angle'],'C':float(c),'P':float(p),'A':None if a is None else float(a),
                   'score':float(score),'alarm':bool(score>threshold),'threshold':threshold,'trace':trace,'x':x,'z':z[0]}
        return self.last
