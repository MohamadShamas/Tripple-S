"""
Copyright (c) 2006-2007, NIPY Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NIPY Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Utility functions and definitions from NIPY package.
"""


from struct import calcsize, pack, unpack
import sys
import os
import numpy as N


######## STRUCT PACKING/UNPACKING/INTERPRETATION ROUTINES ####################
def numvalues(format):
    """ The number of values for the given format. 
    
    :Parameters:
        `format` : TODO
            TODO
            
    :Returns: int
    """
    numstr, fmtchar = format[:-1], format[-1]
    return (numstr and fmtchar not in ("s","p")) and int(numstr) or 1

def elemtype(format):
    """
    Find the type of a given format string

    :Parameters:
        `format` : string
            A format string

    :Returns:
        ``float`` or ``int`` or ``str``

    :Raises ValueError: if an invalid format character is given.
    """
    fmtchar = format[-1]
    for formats, typ in _typemap.items():
        if fmtchar in formats:
            return typ
    raise ValueError("format char %s must be one of: %s"%\
                     (fmtchar, _allformats))

def sanevalues(format, value):
    """
    :Parameters:
        `format` : TODO
            TODO
        `value` : TODO
            TODO
    
    :Returns: ``bool``
    """
    nvals, valtype = isinstance(value, (tuple, list)) and \
                     (len(value), type(value[0])) or (1, type(value))
    
    return elemtype(format) == valtype and numvalues(format) == nvals

def formattype(format):
    """
    :Parameters:
        `format` : TODO
            TODO

    :Returns: TODO
    """
    return numvalues(format) > 1 and list or elemtype(format)

def flatten_values(valseq):
    """ Flattens the type of header values constructed by aggregate. 
    
    :Parameters:
        `valseq` : TODO
            TODO
    
    :Returns: TODO
    """
    if not isinstance(valseq, list):
        return [valseq]
    if valseq == []:
        return valseq
    return flatten_values(valseq[0]) + flatten_values(valseq[1:])

def takeval(numvals, values):
    """ Take numvals from values.

    :Parameters:
        `numvals` : TODO
            TODO
        `values` : TODO
            TODO

    :Returns: a single value if numvals == 1 or else a list of values.
    """
    if numvals == 1:
        return values.pop(0)
    else:
        return [values.pop(0) for _ in range(numvals)]

def struct_format(byte_order, elements):
    """
    :Parameters:
        `byte_order` : ``string``
            TODO
        `elements` : ``[string]``
            TODO
    
    :Returns: ``string``
    """
    return byte_order+" ".join(elements)
   
def aggregate(formats, values):
    """
    :Parameters:
        `formats` : TODO
            TODO
        `values` : TODO
            TODO
            
    :Returns: TODO
    """
    return [takeval(numvalues(format), values) for format in formats]

def struct_unpack(infile, byte_order, elements):
    """
    :Parameters:
        `inflie` : TODO
            TODO
        `byte_order` : TODO
            TODO
        `elements` : TODO
            TODO
            
    :Returns: TODO
    """
    format = struct_format(byte_order, elements)
    return aggregate(elements,
      list(unpack(format, infile.read(calcsize(format)))))

def struct_pack(byte_order, elements, values):
    """
    :Parameters:
        `byte_order` : string
            The byte order to use. Must be one of NATIVE, BIG_ENDIAN,
            LITLE_ENDIAN
        `elements` : [string]
            A list of format string elements to use
        `value` : [ ... ]
            A list of values to be packed into the format string

    :Returns: ``string``
    """
    format = struct_format(byte_order, elements)
    return pack(format, *flatten_values(values))

def touch(filename):
    """ Ensure that filename exists and is writable.

    :Parameters:
        `filename` : string
            The file to be touched

    :Returns: ``None``
    """

    try:
        open(filename, 'a').close()
    except IOError:
        pass
    os.utime(filename, None)


def scale_data(data, new_dtype, default_scale):
    """ Scales numbers in data to desired match dynamic range of new dtype 
    
    :Parameters:
        `data` : TODO
            TODO
        `new_dtype` : TODO
            TODO
        `default_dtype` : TODO
            TODO
            
    :Returns: TODO
    """
    # if casting to an integer type, check the data range
    # if it clips, then scale down
    # if it has poor integral resolution, then scale up
    if new_dtype in _integer_ranges.keys():
        maxval = abs(data.max())
        maxrange = _integer_ranges[new_dtype.type]
        scl = maxval/maxrange or 1.
        return scl
    return default_scale