import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import hashlib
import time
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import platform
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# -----------------------------
# Self-Attention
# -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H*W).permute(0,2,1)
        proj_key = self.key(x).view(B, -1, H*W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(B, -1, H*W)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(B,C,H,W)
        return self.gamma*out + x

# -----------------------------
# Enhanced CNN
# -----------------------------
class DeepfakeDetectorCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.spectral_branch = self._make_branch(3,64)
        self.mfcc_branch = self._make_branch(3,64)
        self.phase_branch = self._make_branch(1,64)
        self.spectral_attention = SelfAttention(256)
        self.mfcc_attention = SelfAttention(256)
        self.phase_attention = SelfAttention(256)
        self.fusion_conv = nn.Conv2d(768,256,1)
        self.fusion_bn = nn.BatchNorm2d(256)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128,num_classes)
        )
    def _make_branch(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(out_channels,out_channels*2,stride=2),
            ResidualBlock(out_channels*2,out_channels*4,stride=2)
        )
    def forward(self, spectral, mfcc, phase):
        spec_out = self.spectral_branch(spectral)
        mfcc_out = self.mfcc_branch(mfcc)
        phase_out = self.phase_branch(phase)
        spec_out = self.spectral_attention(spec_out)
        mfcc_out = self.mfcc_attention(mfcc_out)
        phase_out = self.phase_attention(phase_out)
        target_size = spec_out.shape[2:]
        mfcc_out = F.interpolate(mfcc_out,size=target_size,mode='bilinear',align_corners=False)
        phase_out = F.interpolate(phase_out,size=target_size,mode='bilinear',align_corners=False)
        fused = torch.cat([spec_out,mfcc_out,phase_out],dim=1)
        fused = F.relu(self.fusion_bn(self.fusion_conv(fused)))
        pooled = self.global_pool(fused).view(fused.size(0),-1)
        return self.classifier(pooled)

# -----------------------------
# Lightweight CNN
# -----------------------------
class LightweightDetector(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels,32,3,2,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,64,3,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64,num_classes)
        )
    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_pool(feat).view(feat.size(0),-1)
        return self.classifier(pooled)

# -----------------------------
# Ensemble Detector
# -----------------------------
class EnsembleDetector(nn.Module):
    def __init__(self, models, strategy='standard', weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        if weights is None:
            self.weights = torch.ones(len(models))/len(models)
        else:
            self.weights = torch.tensor(weights,dtype=torch.float32)
            self.weights = self.weights/self.weights.sum()
    def forward_standard(self, spectral, mfcc=None, phase=None):
        probs_list=[]
        for i,model in enumerate(self.models):
            if isinstance(model,LightweightDetector):
                out=model(spectral)
            else:
                out=model(spectral,mfcc,phase)
            probs_list.append(F.softmax(out,dim=1)*self.weights[i])
        return torch.stack(probs_list,dim=0).sum(dim=0)
    def forward_multiscale(self, spectral, mfcc=None, phase=None):
        scales=[1.0,0.75,0.5]
        probs_list=[]
        B,C,H,W=spectral.shape
        for scale in scales:
            new_H=int(H*scale)
            new_W=int(W*scale)
            spec_resized=F.interpolate(spectral,size=(new_H,new_W),mode='bilinear',align_corners=False)
            mfcc_resized=F.interpolate(mfcc,size=(new_H,new_W),mode='bilinear',align_corners=False) if mfcc is not None else None
            phase_resized=F.interpolate(phase,size=(new_H,new_W),mode='bilinear',align_corners=False) if phase is not None else None
            probs_list.append(self.forward_standard(spec_resized,mfcc_resized,phase_resized))
        return torch.stack(probs_list,dim=0).mean(dim=0)
    def forward_adaptive(self, spectral, mfcc=None, phase=None):
        mean_val=spectral.mean(dim=[1,2,3],keepdim=True)
        weights=self.weights.clone()
        if len(weights)==2:
            weights[0]=(0.5+mean_val.squeeze())
            weights[1]=1.0-weights[0]
            weights=weights/weights.sum()
        probs_list=[]
        for i,model in enumerate(self.models):
            if isinstance(model,LightweightDetector):
                out=model(spectral)
            else:
                out=model(spectral,mfcc,phase)
            probs_list.append(F.softmax(out,dim=1)*weights[i])
        return torch.stack(probs_list,dim=0).sum(dim=0)
    def forward(self,spectral,mfcc=None,phase=None):
        if self.strategy=='standard':
            return self.forward_standard(spectral,mfcc,phase)
        elif self.strategy=='multiscale':
            return self.forward_multiscale(spectral,mfcc,phase)
        elif self.strategy=='adaptive':
            return self.forward_adaptive(spectral,mfcc,phase)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")
    def predict(self,spectral,mfcc=None,phase=None):
        probs=self.forward(spectral,mfcc,phase)
        return torch.argmax(probs,dim=1)

# -----------------------------
# Model Factory
# -----------------------------
def get_model(model_type='enhanced', in_channels=1, num_classes=2):
    if model_type=='enhanced':
        return DeepfakeDetectorCNN(num_classes)
    elif model_type=='lightweight':
        return LightweightDetector(in_channels,num_classes)
    elif model_type=='pytorch_model':
        # ResNet + GRU + Attention model (num_classes=1 to match checkpoint)
        from resnet_gru_model import ResNetGRUModel
        return ResNetGRUModel(num_classes=1)
    else:
        return DeepfakeDetectorCNN(num_classes)

# -----------------------------
# Model Optimization Utilities
# -----------------------------
class ModelOptimizer:
    def quantize_dynamic(self,model):
        model.eval()
        return torch.quantization.quantize_dynamic(model,{nn.Linear,nn.Conv2d},dtype=torch.qint8)
    def prune(self,model,ratio=0.2):
        import torch.nn.utils.prune as prune
        parameters_to_prune=[]
        for _,m in model.named_modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                parameters_to_prune.append((m,'weight'))
        prune.global_unstructured(parameters_to_prune,pruning_method=prune.L1Unstructured,amount=ratio)
        for m,p in parameters_to_prune:
            prune.remove(m,p)
        return model
    def compile(self,model):
        try:
            return torch.compile(model,mode='reduce-overhead') if platform.system()!='Windows' else torch.compile(model,mode='reduce-overhead',fullgraph=False)
        except:
            return model

# -----------------------------
# Feature Caching
# -----------------------------
class FeatureCache:
    def __init__(self,cache_dir='./feature_cache',max_size=1000):
        self.cache_dir=cache_dir
        self.max_size=max_size
        self.memory_cache={}
        os.makedirs(cache_dir,exist_ok=True)
    def _hash_file(self,path):
        with open(path,'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    def _cache_path(self,file_hash):
        return os.path.join(self.cache_dir,f"{file_hash}.pkl")
    def get(self,path):
        h=self._hash_file(path)
        if h in self.memory_cache:
            return self.memory_cache[h]
        p=self._cache_path(h)
        if os.path.exists(p):
            try:
                with open(p,'rb') as f:
                    feat=pickle.load(f)
                if len(self.memory_cache)<self.max_size:
                    self.memory_cache[h]=feat
                return feat
            except:
                pass
        return None
    def cache(self,path,features):
        h=self._hash_file(path)
        if len(self.memory_cache)<self.max_size:
            self.memory_cache[h]=features
        try:
            with open(self._cache_path(h),'wb') as f:
                pickle.dump(features,f)
        except:
            pass
    def clear(self):
        self.memory_cache.clear()
        for file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir,file))

# -----------------------------
# Parallel Processing
# -----------------------------
class ParallelProcessor:
    def __init__(self,num_workers=None):
        self.num_workers=num_workers or mp.cpu_count()
    def parallel_extract(self,paths,func):
        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            futures=[ex.submit(func,p) for p in paths]
            return [f.result() for f in futures]

# -----------------------------
# Profiler
# -----------------------------
class Profiler:
    def __init__(self):
        self.times={}
    def profile(self,name):
        def decorator(func):
            def wrapper(*args,**kwargs):
                start=time.time()
                result=func(*args,**kwargs)
                end=time.time()
                self.times.setdefault(name,[]).append(end-start)
                return result
            return wrapper
        return decorator
    def report(self):
        for k,v in self.times.items():
            print(f"{k}: avg {np.mean(v):.4f}s, min {np.min(v):.4f}s, max {np.max(v):.4f}s, calls {len(v)}")

# -----------------------------
# Example: create optimized model
# -----------------------------
def create_optimized_model(model,level='medium'):
    opt=ModelOptimizer()
    if level=='light':
        return opt.compile(model)
    elif level=='medium':
        m=opt.quantize_dynamic(model)
        return opt.compile(m)
    elif level=='heavy':
        m=opt.quantize_dynamic(model)
        m=opt.prune(m,0.1)
        return opt.compile(m)
    else:
        return model
