
import sys
from prep import prep
from train import train


def main(mode):
    if mode == 'prep':
        prep()
    if mode == 'train':
        train()

if __name__ == '__main__':
    main(sys.argv[1])
