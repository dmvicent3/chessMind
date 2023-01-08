import random
import shutil
import glob

for c in random.sample(glob.glob('./data/train/bb/*.JPG'), 200):
    shutil.move(c, './data/test/bb')
    
for c in random.sample(glob.glob('./data/train/bk/*.JPG'), 200):
    shutil.move(c, './data/test/bk')
    
for c in random.sample(glob.glob('./data/train/bn/*.JPG'), 200):
    shutil.move(c, './data/test/bn')
    
for c in random.sample(glob.glob('./data/train/bp/*.JPG'), 600):
    shutil.move(c, './data/test/bp')
    
for c in random.sample(glob.glob('./data/train/bq/*.JPG'), 200):
    shutil.move(c, './data/test/bq')
    
for c in random.sample(glob.glob('./data/train/br/*.JPG'), 200):
    shutil.move(c, './data/test/br')
    
for c in random.sample(glob.glob('./data/train/empty/*.JPG'), 1000):
    shutil.move(c, './data/test/empty')
    
for c in random.sample(glob.glob('./data/train/wb/*.JPG'), 200):
    shutil.move(c, './data/test/wb')
    
for c in random.sample(glob.glob('./data/train/wk/*.JPG'), 200):
    shutil.move(c, './data/test/wk')
    
for c in random.sample(glob.glob('./data/train/wn/*.JPG'), 200):
    shutil.move(c, './data/test/wn')
    
for c in random.sample(glob.glob('./data/train/wp/*.JPG'), 600):
    shutil.move(c, './data/test/wp')
    
for c in random.sample(glob.glob('./data/train/wq/*.JPG'), 200):
    shutil.move(c, './data/test/wq')
    
for c in random.sample(glob.glob('./data/train/wr/*.JPG'), 200):
    shutil.move(c, './data/test/wr')