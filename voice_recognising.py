import numpy as np
import torch
import math
from pathlib import Path
from encoder import inference as encoder
from synthesizer.inference import Synthesizer

"""
To find out if two voices from different sound files are different:
1) get the embeding of both sound files. The embedings are vectors.
2) find the euclidean distance of the embedings.
3) compare the distance to some number, 0.85 is suggested
4) if it is less than, it is the same voice
5) if it is greater than, it is a different voice
"""

differentDistance=0.85#the distance it has to be to be considered a different voice

#finds the distance between two vectors, used for the distance between embedings in this case
def find_distance(emb1, emb2):
    if(len(emb1)!=len(emb2)):
        print("Error: different dimensions")
        return
    else:
        total=0
        for i in range(len(emb1)):
            total+=math.pow(emb1[i]-emb2[i],2)
        return math.sqrt(total)

#checks if two embedings reference the same voice
def isSameVoice(emb1, emb2):
    return find_distance(emb1, emb2) <=  differentDistance

#checks if two embedings reference different voices
def isDifferentVoice(emb1, emb2):
    return find_distance(emb1, emb2) >  differentDistance
 
#gets wave from file path and name
def getWavFromFile(path):
    return Synthesizer.load_preprocess_wav(Path(path))
    
#get embeding from wave
def getEmbedingFromWav(wav):
    return encoder.embed_utterance(wav)

encoder_path="encoder/saved_models/pretrained.pt"
encoder.load_model(Path(encoder_path))

#test program
if(__name__ =="__main__"):
    sound1_path=input("Enter sound file 1 ")
    sound2_path=input("Enter sound file 2 ")
    wav1 = getWavFromFile(sound1_path)
    print("loaded wav 1")
    print(wav1)

    wav2 = getWavFromFile(sound2_path)
    print("loaded wav 2")
    print(wav2)


    embed1=getEmbedingFromWav(wav1)
    print("embeded1")
    print(embed1)

    embed2=getEmbedingFromWav(wav2)
    print("embeded2")
    print(embed2)

    distance=find_distance(embed1, embed2)
    if(distance!=None):
        print("found distance!")
        print(distance)
        if(distance>differentDistance):#if different voice
            print("Different voice!")
        else:
            print("Same voice!")
