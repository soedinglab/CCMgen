import ccmpred
import sys


def logo(color=None):
    version = ccmpred.__version__

    if color is None:
        color = (sys.stdin.isatty()) and (sys.stdout.isatty())

    if color:
        print(" _____ _____ _____               _")
        print("|\x1b[30;42m     |     |     |\x1b[0m___ ___ ___ _|\x1b[30;44m |\x1b[0m")
        print("|\x1b[30;42m   --|   --| | | |\x1b[44m . |  _| -_| . |\x1b[0m")
        print("|\x1b[30;42m_____|_____|_|_|_|\x1b[44m  _|_|\x1b[0m \x1b[30;44m|___|___|\x1b[0m version {0}".format(version))
        print("                  |\x1b[30;44m_|\x1b[0m\n")
    else:
        print(" _____ _____ _____               _")
        print("|     |     |     |___ ___ ___ _| |")
        print("|   --|   --| | | | . |  _| -_| . |")
        print("|_____|_____|_|_|_|  _|_| |___|___| version {0}".format(version))
        print("                  |_|\n")
