import msgpack
import functools
import numpy as np
import re
import json
import gzip
from six import string_types, StringIO


META_PREFIX = "#>META> "


class CCMRaw(object):
    """Storage class for CCMpred raw prediction"""
    def __init__(self, ncol, x_single, x_pair, meta):
        self.ncol = ncol
        self.x_single = x_single
        self.x_pair = x_pair
        self.meta = meta

    def __repr__(self):
        return "<CCMRaw ncol={0}>".format(self.ncol)


def stream_or_file(mode='r'):
    """Decorator for making a function accept either a filename or file-like object as a first argument"""

    def inner(fn):
        @functools.wraps(fn)
        def streamify(f, *args, **kwargs):
            if isinstance(f, string_types):

                open_fn = gzip.open if f.endswith(".gz") else open

                try:
                    fh = open_fn(f, mode)
                    res = fn(fh, *args, **kwargs)
                finally:
                    fh.close()

                return res
            else:
                return fn(f, *args, **kwargs)

        return streamify

    return inner


_PARSERS = []


def parser(fn):
    _PARSERS.append(fn)
    return fn


@parser
@stream_or_file('rb')
def parse_msgpack(f):
    """Parse a msgpack CCMpred prediction from a filename or file object"""
    x = msgpack.unpackb(f.read(), encoding="utf-8")

    assert(x['format'] == 'ccm-1')

    ncol = x['ncol']
    x_single = np.array(x['x_single']).reshape((ncol, 20))
    x_pair = np.zeros((ncol, ncol, 21, 21))

    meta = x['meta'] if 'meta' in x else None

    for p in x['x_pair'].values():
        i = p['i']
        j = p['j']
        mat = np.array(p['x']).reshape((21, 21))
        x_pair[i, j, :, :] = mat
        x_pair[j, i, :, :] = mat.T

    return CCMRaw(ncol, x_single, x_pair, meta)


@parser
@stream_or_file('r')
def parse_oldraw(f):
    """Read raw emission potentials from rawfile"""

    buf = StringIO()
    re_identifier = re.compile("^#\s*(\d+)\s+(\d+)\s*$")

    x_single = None
    x_pair = None
    i, j = None, None
    meta = None
    for line_idx, line in enumerate(f):
        if line.startswith(META_PREFIX):
            meta = json.loads(line[len(META_PREFIX):].strip())

        elif line.startswith("#"):

            buf.seek(0)

            if x_single is not None:
                x_pair[i, j, :, :] = np.loadtxt(buf)
                x_pair[j, i, :, :] = x_pair[i, j, :, :].T

            else:
                x_single = np.loadtxt(buf)

                ncol = x_single.shape[0]
                x_pair = np.zeros((ncol, ncol, 21, 21))

            buf = StringIO()

            m = re_identifier.match(line)
            if m:
                i, j = int(m.group(1)), int(m.group(2))

            else:
                raise Exception("Line {0} starts with # but doesn't match regex!".format(line_idx + 1))

        else:
            buf.write(line)

    if x_single is not None and buf.tell():
        buf.seek(0)
        x_pair[i, j, :, :] = np.loadtxt(buf)
        x_pair[j, i, :, :] = x_pair[i, j, :, :].T

    return CCMRaw(ncol, x_single, x_pair, meta)


def parse(f):
    r = None
    for parser in _PARSERS:
        try:
            if hasattr(f, 'seek'):
                f.seek(0)

            r = parser(f)
        except Exception as e:
            pass

        if r is not None:
            continue
    return r


@stream_or_file('wb')
def write_msgpack(f, data):

    x_single = data.x_single.reshape(data.ncol * 20).tolist()
    x_pair = {}
    for i in range(data.ncol):
        for j in range(i + 1, data.ncol):
            x_pair["{0}/{1}".format(i, j)] = {
                "i": i,
                "j": j,
                "x": data.x_pair[i, j, :, :].reshape(21 * 21).tolist()
            }

    out = {
        "format": "ccm-1",
        "ncol": data.ncol,
        "x_single": x_single,
        "x_pair": x_pair
    }

    if data.meta:
        out['meta'] = data.meta

    f.write(msgpack.packb(out))


@stream_or_file('wb')
def write_oldraw(f, data):
    np.savetxt(f, data.x_single, delimiter="\t")

    for i in range(data.ncol):
        for j in range(i + 1, data.ncol):
            f.write("# {0} {1}\n".format(i, j).encode("utf-8"))
            np.savetxt(f, data.x_pair[i, j], delimiter="\t")

    if data.meta:
        f.write(META_PREFIX.encode("utf-8") + json.dumps(data.meta).encode("utf-8") + b"\n")

if __name__ == '__main__':
    # data = parse_oldraw("data/test.raw")
    data = parse_msgpack("data/test.braw")

    print("data:")
    print(data)

    print("data.x_single.shape:")
    print(data.x_single.shape)

    print("data.x_single:")
    print(data.x_single)

    print("data.x_pair.shape:")
    print(data.x_pair.shape)

    print("data.x_pair[3, 4]:")
    print(data.x_pair[3, 4])
