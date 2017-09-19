
import sys
from prep import prep


def main(mode):
    if mode == 'prep':
        prep()


if __name__ == '__main__':
    main(sys.argv[1])
