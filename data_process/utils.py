

def write_log(*t, path="./log.txt"):
    t = ",".join([str(item) for item in t])
    f = open(path, "a")
    f.write(t + '\n')
    f.close()


def analyzing_asr_impact(pre_results, label_path, trans_path, emo2idx, idx2emo):

    
    