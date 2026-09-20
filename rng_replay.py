"""Replay full RNG streams, never approximate a continuous stream with seed+i."""
import hashlib,random
import numpy as np
import torch

def capture():
 return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),'cuda':[s.clone() for s in torch.cuda.get_rng_state_all()]}

def restore(state):
 random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])

def fingerprint(state):
 h=hashlib.sha256();h.update(repr(state['python']).encode());n=state['numpy'];h.update(n[0].encode());h.update(n[1].tobytes());h.update(repr(n[2:]).encode());h.update(state['torch'].numpy().tobytes())
 for s in state['cuda']:h.update(s.numpy().tobytes())
 return h.hexdigest()

def tensor_hash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
