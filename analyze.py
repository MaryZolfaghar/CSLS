# TODO:
#   -Episodic analysis: hub retrieval
#   -Cortical analysis: PCA, correlation

import torch 
import numpy as np

def analyze(meta, model, loader, args):
    if meta:
        results = analyze_hubs(model, loader)
    else:
        pca_results = analyze_pca(model, loader)
        correlation_results = analyze_r(model, loader)
        results = {'pca':pca_results, 'corr':correlation_results}
    return results 

def analyze_hubs(model, loader):
    return "Not implemented"

def analyze_pca(model, loader):
    return "Not implemented"

def analyze_r(model, loader):
    return "Not implemented"