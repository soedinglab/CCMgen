#!/usr/bin/env python
"""Convert a msgpack potential file to flatfile format"""

import ccmraw as cr


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_msgpack", help="Input raw file in new msgpack format")
    parser.add_argument("out_flat", help="Output raw file in old flatfile format")

    opt = parser.parse_args()

    cr.write_oldraw(opt.out_flat, cr.parse_msgpack(opt.in_msgpack))


if __name__ == '__main__':
    main()
