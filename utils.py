import os

def Search(abspath='.'):
    os.chdir(abspath)
    L = os.listdir('.')
    filenames=[]
    for v in L:
        if os.path.isfile(v) and 'acc' in v:
            filenames.append(v)
    return filenames



def all_files_under_deletetemp(path, sort=True):
    if path != None:
            #filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
            filenames  = Search(os.path.join(path))
    if sort:
        filenames = sorted(filenames)
    return filenames

def all_files_under(path, sort=True):
    if path != None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
    if sort:
        filenames = sorted(filenames)
    return filenames
