import sys, os, argparse
import utils
import numpy as np

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Extract images with valid range (between -99 and 99 ).')
    parser.add_argument('--root', dest='root', help='Path of AFLW2000')
    args = parser.parse_args()
    return args

def DB_AFLW2000Iterator(path):
    extensions = ('.jpg')
    f = open(os.path.join(path, 'filenames.txt'), 'w+')
    counter = 0
    for root, dirs, files in os.walk(path):
        for file_ in files:
            names = os.path.splitext(file_)
            if names[1] in extensions:
                #print( os.path.join(root, file_) )
                names = os.path.splitext(file_)
                mat_path = os.path.join(root, names[0]+'.mat')
                # We get the pose in radians
                pose = utils.get_ypr_from_mat(mat_path)
                # And convert to degrees.
                pitch = pose[0] * 180 / np.pi
                yaw = pose[1] * 180 / np.pi
                roll = pose[2] * 180 / np.pi
                if pitch >= -99.0 and pitch <= 99.0 and yaw >= -99.0 and yaw <= 99.0 and roll >= -99.0 and roll <= 99.0:
                    print(mat_path)
                    counter = counter +1
                    f.write(names[0]+os.linesep)
    print counter
    f.close()

def DB_300W_LP_Iterator(path):
    extensions = ('.jpg')
    subfolders = ['IBUG', 'AFW', 'LFPW', 'IBUG_Flip', 'AFW_Flip', 'HELEN', 'LFPW_Flip', 'HELEN_Flip']
    f = open(os.path.join(path, 'filenames.txt'), 'w+')
    counter = 0
    for s in subfolders:
        root = os.path.join(path, s)
        files = os.listdir(root)

        for file_ in files:
            names = os.path.splitext(file_)
            if names[1] in extensions:
                names = os.path.splitext(file_)
                mat_path = os.path.join(root, names[0]+'.mat')
                # We get the pose in radians
                pose = utils.get_ypr_from_mat(mat_path)
                # And convert to degrees.
                pitch = pose[0] * 180 / np.pi
                yaw = pose[1] * 180 / np.pi
                roll = pose[2] * 180 / np.pi
                if pitch >= -99.0 and pitch <= 99.0 and yaw >= -99.0 and yaw <= 99.0 and roll >= -99.0 and roll <= 99.0:
                    counter = counter +1
                    f.write(s + os.sep+ names[0]+os.linesep)
    print counter
    f.close()

if __name__ == '__main__':
    args = parse_args()
    #DB_AFLW2000Iterator(args.root)
    DB_300W_LP_Iterator(args.root)
