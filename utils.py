

def write_log(*t, path="./log.txt"):
    t = ",".join([str(item) for item in t])
    f = open(path, "a")
    f.write(t + '\n')
    f.close()