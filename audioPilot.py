"""
audioPilot - automatically finding the best position for transition between songs by content-based audio analysis.
    Under development by Cheng-i Wang
    contacts:
        wangsix@gmail.com
        
Required modules:
    Numpy
    Scipy
    Echonest Remix API (with API key)

"""
from echonest.remix import audio #@UnresolvedImport
import dirac #@UnresolvedImport 
import sys 
import numpy as np
import scipy.stats.mstats as mstats
import scipy.signal as signal 

USAGE = """
Usage:
    python audioPilot.py <input_filename> <input2_filename> <output_filename> <# of beats>
Exampel:
    python audioPilot.py "CryMeARiver.wav" "CryMeARiver2.wav" "CryCycle.wav" 32 
"""

def low_pass(vec, ratio):
    """ Utility function for simple low-pass filtering
    
    Args:
        vec: The vector containing the signal being low-pass filtered.
        ratio: The ratio (0~1) for feed-back coefficient. The closer to 1 the smooth the curve. 
    """
    vec = signal.lfilter([ratio],[1.0, -(1.0 - ratio)], vec)
    vec = signal.lfilter([ratio],[1.0, -(1.0 - ratio)], vec[-1::-1])
    vec = vec[-1::-1]
    return vec

def analyze(beats, segments):
    beat_start = [x.start for x in beats]
    seg_start = np.array([x.start for x in segments])
    seg_dur = np.array([x.duration for x in segments])
    beat_con = [x.confidence for x in beats]
    
    seg_mel = np.array([x.timbre for x in segments]).transpose()
    seg_mel = (-seg_mel[1]*seg_mel[2]*seg_mel[3]).transpose()
    seg_mel = np.abs(np.min(seg_mel)) + seg_mel
    seg_mel = seg_mel/np.max(seg_mel)
    
    syn_mel = np.zeros((len(beats),))
    syn_ch = np.zeros((len(beats),))
    
    seg_ch = np.array([x.pitches for x in segments])
    seg_ch = mstats.gmean(seg_ch, axis = 1)/np.mean(seg_ch, axis = 1)
        
    _last = 0
    for i in range(len(beats)):
        _tmp_ind = np.argmin(np.abs(seg_start - beat_start[i]))
        if _tmp_ind  < len(seg_dur):
            syn_mel[i] = np.dot(seg_dur[_last:_tmp_ind+1],
                                seg_mel[_last:_tmp_ind+1])/np.sum(seg_dur[_last:_tmp_ind+1])
            syn_ch[i] = np.dot(seg_dur[_last:_tmp_ind+1],
                               seg_ch[_last:_tmp_ind+1])/np.sum(seg_dur[_last:_tmp_ind+1])
        else:
            syn_mel[i] = np.dot(seg_dur[_last:],seg_mel[_last:])/np.sum(seg_dur[_last:])
            syn_ch[i] = np.dot(seg_dur[_last:],seg_ch[_last:])/np.sum(seg_dur[_last:])
        _last = _tmp_ind 
    
    beat_val = syn_mel*beat_con
    beat_val = low_pass(beat_val, 0.8)    
    harm_val = low_pass(syn_ch, 0.8)
    return beat_val, harm_val

def locate(a, b, win):
    c_a = signal.convolve(a, win, mode = 'valid')
    c_b = signal.convolve(b, win, mode = 'valid')
    val = np.max(c_a)*np.max(c_b)
    ind_a = np.argmax(c_a) + len(win)/2
    ind_b = np.argmax(c_b) + len(win)/2
    return val, ind_a, ind_b

def pre_process(input_file):
    _f = audio.LocalAudioFile(input_file)
    _b = _f.analysis.beats
    _s = _f.analysis.segments
    _beat, _harm = analyze(_b,_s)
    return _f, _b, _beat, _harm

def beat_process(audioData, beats, ind, win, ratio, i, vol):
    vecin = audioData[beats[ind-(win/2)+i]].data
    vecout = dirac.timeScale(vecin, ratio, audioData.sampleRate, 0)
    new = audio.AudioData(ndarray=vecout, shape=vecout.shape, 
                          sampleRate=audioData.sampleRate, 
                          numChannels=vecout.shape[1])
    a = audio.AmplitudeFactor(vol[i])
    new = a.modify(new)
    return new

def main():
    try:
        in_file1 = sys.argv[1]
        in_file2 = sys.argv[2]
        out_file = sys.argv[3]
        win_l = sys.argv[4]
    except Exception:
        print USAGE
        sys.exit(-1)
        
    f, b, beat1, harm1 = pre_process(in_file1)
    f2, b2, beat2, harm2 = pre_process(in_file2)

    winLength = int(win_l)
    window = signal.gaussian(winLength, 1)
    
    v1, x1, x2 = locate(beat1, harm2, window)
    v2, y2, y1 = locate(beat2, harm1, window)
    if v1 > v2:
        ind1 = x1
        ind2 = x2
    else:
        ind1 = y1
        ind2 = y2
    start_t = (60.0/f.analysis.tempo['value'])
    end_t = (60.0/f2.analysis.tempo['value'])
    dur = np.linspace(start_t, end_t, winLength)
    vol1 = np.power(np.linspace(1,0,winLength),1.0/2.0)
    vol2 = np.power(np.linspace(0,1,winLength),1.0/2.0)
    collect = []
    for i in range(winLength):
        ratio1 = dur[i]/start_t
        ratio2 = dur[i]/end_t
        new1 = beat_process(f, b, ind1, winLength, ratio1, i, vol1)
        new2 = beat_process(f2, b2, ind2, winLength, ratio2, i, vol2)
        new1.sum(new2)
        collect.append(new1)
    out = audio.assemble(collect, numChannels = 2)
    c1 = []
    c2 = []
    for j in range(8):
        n1 = f[b[ind1-(winLength/2+8)+j]]
        n2 = f2[b2[ind2+(winLength/2)+j]]
        c1.append(n1)
        c2.append(n2)
    o1 = audio.assemble(c1, numChannels = 2)
    o2 = audio.assemble(c2, numChannels = 2)
    o1.append(out)
    o1.append(o2)
    o1.encode(out_file)
    
if __name__ == '__main__':
    main()