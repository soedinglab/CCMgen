# coding: utf-8
import ccmpred
import sys

is_tty = (sys.stdin.isatty()) and (sys.stdout.isatty())

LOGOS = {}
LOGOS['ccmpred', True] = """
  \x1b[32m┏━╸┏━╸┏┳┓\x1b[34m┏━┓┏━┓┏━╸╺┳┓\x1b[0m  version {0}
  \x1b[32m┃  ┃  ┃┃┃\x1b[34m┣━┛┣┳┛┣╸  ┃┃\x1b[0m  Seemayer and Soeding (2016)
  \x1b[32m┗━╸┗━╸╹ ╹\x1b[34m╹  ╹┗╸┗━╸╺┻┛\x1b[0m  https://github.com/soedinglab
"""

LOGOS['ccmpred', False] = """
  ┏━╸┏━╸┏┳┓┏━┓┏━┓┏━╸╺┳┓  version {0}
  ┃  ┃  ┃┃┃┣━┛┣┳┛┣╸  ┃┃  Seemayer and Soeding (2016)
  ┗━╸┗━╸╹ ╹╹  ╹┗╸┗━╸╺┻┛  https://github.com/soedinglab
"""


LOGOS['ccmgen', True] = """
  \x1b[32m┏━╸┏━╸┏┳┓\x1b[34m┏━╸┏━╸┏┓╻\x1b[0m  version {0}
  \x1b[32m┃  ┃  ┃┃┃\x1b[34m┃╺┓┣╸ ┃┗┫\x1b[0m  Seemayer and Soeding (2016)
  \x1b[32m┗━╸┗━╸╹ ╹\x1b[34m┗━┛┗━╸╹ ╹\x1b[0m  https://github.com/soedinglab
"""

LOGOS['ccmgen', False] = """
  ┏━╸┏━╸┏┳┓┏━╸┏━╸┏┓╻  version {0}
  ┃  ┃  ┃┃┃┃╺┓┣╸ ┃┗┫  Seemayer and Soeding (2016)
  ┗━╸┗━╸╹ ╹┗━┛┗━╸╹ ╹  https://github.com/soedinglab
"""



def logo(what_for="ccmpred", color=is_tty):
    version = ccmpred.__version__

    print(LOGOS[what_for, color].format(version))
