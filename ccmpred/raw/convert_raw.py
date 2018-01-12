#!/usr/bin/env python
"""Convert a raw potential file to msgpack format"""

import ccmraw as cr


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_raw", help="Input raw file in old raw format")
    parser.add_argument("out_msgpack", help="Output raw file in new msgpack format")

    opt = parser.parse_args()

    cr.write_msgpack(opt.out_msgpack, cr.parse_oldraw(opt.in_raw))


if __name__ == '__main__':
    main()
