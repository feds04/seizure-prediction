import csv
import os.path
import scipy.io as sio
import scipy.stats
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
import numpy as np
from math import log

def compute_features(data, sfreq): # the default input is (n, 239766L) array
# which  contains measurements for n channels, 10 minutes at 400 Hz

    num_channels=data.shape[0] # we have 16 channels for Dogs1-4 and 15 for Dog5
    channels_to_use=range(num_channels)+[0]*(16-num_channels) # we duplicate the first channel for Dog5
    data_len=data.shape[1]
    block_size=data_len/10
    nd1 = block_size/sfreq ## number of data points per 1 Hz
    features=[]
    for j in range(10): # split each block into ten 1 min blocks
      # eigenvalues of the correlation matrix
      # turned out to be of little importance
      ev=np.linalg.eig(np.corrcoef(data[:,j*block_size:(j+1)*block_size]))[0]
      features=np.append(features,ev)
        
        
      for i in channels_to_use:
        block1=data[i,j*block_size:(j+1)*block_size]
        #block1=np.hamming(block_size)*data[i,j*block_size:(j+1)*block_size]
        spectrum=(np.absolute(np.fft.fft(block1))**2)
        delta = sum(spectrum[nd1/10:4*nd1]) # 0.1-4 Hz - delta band
        theta = sum(spectrum[4*nd1:8*nd1]) #  4-8 Hz - theta band
        alpha = sum(spectrum[8*nd1:12*nd1]) #  8-12 Hz
        beta = sum(spectrum[12*nd1:30*nd1]) #  12-30 Hz
        low_gamma = sum(spectrum[30*nd1:70*nd1]) #  30-70 Hz
        high_gamma = sum(spectrum[70*nd1:180*nd1]) #  70-180 Hz
        features=np.append(features,np.log([delta,theta,alpha,beta,low_gamma,high_gamma]))
    return np.reshape(np.array(features),(1,len(features))) # the output is 1 x #(features) array
    

def get_features(subj_name):
    fileno=1

    while True:
      filename=subj_name+"\\"+subj_name+"_interictal_segment"+"_%04d.mat" % (fileno)

    # Assume that the data files are in the current directory
      if not os.path.isfile(filename):
          
         break
      else:
          mat=sio.loadmat(filename)
          print filename
          nd=mat['interictal_segment_'+"%d" % (fileno)]
          freq=round(nd['sampling_frequency'].item(0)[0])
          features=compute_features(nd['data'].item(0), freq)
          
          #print features.shape
          features=np.append(features,np.array([[0]]),axis=1)  # a feature for classification , 0=interictal
          if fileno==1:
              features_array1=features
          else:
              features_array1=np.append(features_array1,features, axis=0)
      fileno +=1
    fileno=1

    while True:
      filename=subj_name+"\\"+subj_name+"_preictal_segment"+"_%04d.mat" % (fileno)
    
      if not os.path.isfile(filename):
         break
      else:
          mat=sio.loadmat(filename)
          print filename
          nd=mat['preictal_segment_'+"%d" % (fileno)]
          freq=round(nd['sampling_frequency'].item(0)[0])
          features=compute_features(nd['data'].item(0), freq)
          features=np.append(features,np.array([[1]]),axis=1)  # 1=preictal
          if fileno==1:
              features_array2=features
          else:
              features_array2=np.append(features_array2,features, axis=0)
      fileno +=1

    return np.append(features_array1, features_array2, axis=0)

def get_test_features(subj_name):
    reader = csv.reader(open('SzPrediction_answer_key.csv', 'r'))
    reader.next() # skip the header
    answer_dict = {}
    for row in reader:
      k, v = row
      answer_dict[k] = int(v)
        
    fileno=1

    while True:
      filename=subj_name+"\\"+subj_name+"_test_segment"+"_%04d.mat" % (fileno)
      keyname=subj_name+"_test_segment"+"_%04d.mat" % (fileno)

    # Assume that the data files are in the current directory
      if not os.path.isfile(filename):
          
         break
      else:
          mat=sio.loadmat(filename)
          print filename
          nd=mat['test_segment_'+"%d" % (fileno)]
          freq=round(nd['sampling_frequency'].item(0)[0])
          features=compute_features(nd['data'].item(0), freq)
          
          #print features.shape
          features=np.append(features,np.array([[answer_dict[keyname]]]),axis=1)  
          # a feature for classification,        0=interictal
          if fileno==1:
              features_array3=features
          else:
              features_array3=np.append(features_array3,features, axis=0)
      fileno +=1

    return features_array3

