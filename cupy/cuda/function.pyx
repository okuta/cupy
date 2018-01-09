# distutils: language = c++

import numpy
import six

cimport cpython
from libcpp cimport vector

from cupy.core cimport core
from cupy.cuda cimport driver
from cupy.cuda cimport runtime
from cupy.cuda cimport stream as stream_module


cdef extern from "cupy_stdint.h" nogil:
    ctypedef signed char int8_t
    ctypedef signed short int16_t
    ctypedef signed int int32_t
    ctypedef signed long long int64_t


cdef class CPointer:
    def __init__(self, p=0):
        self.ptr = <void*>p
        self.size = 0


cdef class CInt8(CPointer):
    cdef:
        int8_t val

    def __init__(self, int8_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = 1


cdef class CInt16(CPointer):
    cdef:
        int16_t val

    def __init__(self, int16_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = 2


cdef class CInt32(CPointer):
    cdef:
        int32_t val

    def __init__(self, int32_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = 4


cdef class CInt64(CPointer):
    cdef:
        int64_t val

    def __init__(self, int64_t v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = 8


cdef class CInt128(CPointer):
    cdef:
        double complex val

    def __init__(self, double complex v):
        self.val = v
        self.ptr = <void*>&self.val
        self.size = 16


cdef set _pointer_numpy_types = {numpy.dtype(i).type
                                 for i in '?bhilqBHILQefdFD'}


cdef inline CPointer _pointer(x):
    cdef Py_ssize_t itemsize
    if x is None:
        return CPointer()
    if isinstance(x, core.ndarray):
        return (<core.ndarray>x).get_pointer()
    if isinstance(x, core.Indexer):
        return (<core.Indexer>x).get_pointer()

    if type(x) not in _pointer_numpy_types:
        if isinstance(x, six.integer_types):
            x = numpy.int64(x)
        elif isinstance(x, float):
            x = numpy.float64(x)
        elif isinstance(x, bool):
            x = numpy.bool_(x)
        else:
            raise TypeError('Unsupported type %s' % type(x))

    itemsize = x.itemsize
    if itemsize == 1:
        return CInt8(x.view(numpy.int8))
    if itemsize == 2:
        return CInt16(x.view(numpy.int16))
    if itemsize == 4:
        return CInt32(x.view(numpy.int32))
    if itemsize == 8:
        return CInt64(x.view(numpy.int64))
    if itemsize == 16:
        return CInt128(x.view(numpy.complex128))
    raise TypeError('Unsupported type %s. (size=%d)', type(x), itemsize)


cdef inline size_t _get_stream(stream) except *:
    if stream is None:
        return stream_module.get_current_stream_ptr()
    else:
        return stream.ptr


cdef void _launch_cuda(size_t func, Py_ssize_t grid0, int grid1, int grid2,
                       Py_ssize_t block0, int block1, int block2,
                       args, Py_ssize_t shared_mem, size_t stream) except *:
    cdef list pargs = []
    cdef vector.vector[void*] kargs
    cdef CPointer cp
    kargs.reserve(len(args))
    for a in args:
        cp = _pointer(a)
        pargs.append(cp)
        kargs.push_back(cp.ptr)

    runtime._ensure_context()
    driver.launchKernel(
        func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
        <int>shared_mem, stream, <size_t>&(kargs[0]), <size_t>0)


cdef void _launch_hip(size_t func, Py_ssize_t grid0, int grid1, int grid2,
                      Py_ssize_t block0, int block1, int block2,
                      args, Py_ssize_t shared_mem, size_t stream) except *:
    cdef vector.vector[unsigned char] kargs
    cdef vector.vector[size_t] config
    cdef CPointer cp
    cdef unsigned char* ptr
    cdef size_t total
    cdef Py_ssize_t size, pad
    kargs.assign(24, 0)
    for a in args:
        cp = _pointer(a)
        ptr = <unsigned char*>cp.ptr
        size = min(cp.size, 8)
        pad = -kargs.size() & (size - 1)
        for i in range(pad):
            kargs.push_back(0)
        kargs.insert(kargs.end(), ptr, ptr + cp.size)
    total = kargs.size()
    # define HIP_LAUNCH_PARAM_BUFFER_POINTER ((void*) 0x01)
    # define HIP_LAUNCH_PARAM_BUFFER_SIZE    ((void*) 0x02)
    # define HIP_LAUNCH_PARAM_END            ((void*) 0x03)
    config.push_back(1)
    config.push_back(<size_t><void*>&(kargs[0]))
    config.push_back(2)
    config.push_back(<size_t><void*>&total)
    config.push_back(3)

    driver.launchKernel(
        func, <int>grid0, grid1, grid2, <int>block0, block1, block2,
        <int>shared_mem, stream, 0, <size_t>&(config[0]))


cdef void _launch(size_t func, Py_ssize_t grid0, int grid1, int grid2,
                  Py_ssize_t block0, int block1, int block2,
                  args, Py_ssize_t shared_mem, size_t stream) except *:
    if runtime._is_hip_environment:
        _launch_hip(func, grid0, grid1, grid2, block0, block1, block2,
                    args, shared_mem, stream)
    else:
        _launch_cuda(func, grid0, grid1, grid2, block0, block1, block2,
                     args, shared_mem, stream)


cdef class Function:

    """CUDA kernel function."""

    def __init__(self, Module module, str funcname):
        self.module = module  # to keep module loaded
        self.ptr = driver.moduleGetFunction(module.ptr, funcname)

    def __call__(self, tuple grid, tuple block, args, size_t shared_mem=0,
                 stream=None):
        grid = (grid + (1, 1))[:3]
        block = (block + (1, 1))[:3]
        s = _get_stream(stream)
        _launch(
            self.ptr,
            max(1, grid[0]), max(1, grid[1]), max(1, grid[2]),
            max(1, block[0]), max(1, block[1]), max(1, block[2]),
            args, shared_mem, s)

    cpdef linear_launch(self, size_t size, args, size_t shared_mem=0,
                        size_t block_max_size=128, stream=None):
        # TODO(beam2d): Tune it
        cdef size_t gridx = min(
            0x7fffffffUL, (size + block_max_size - 1) // block_max_size)
        cdef size_t blockx = min(block_max_size, size)
        s = _get_stream(stream)
        _launch(self.ptr,
                gridx, 1, 1, blockx, 1, 1, args, shared_mem, s)


cdef class Module:

    """CUDA kernel module."""

    def __init__(self):
        self.ptr = 0

    def __del__(self):
        if self.ptr:
            driver.moduleUnload(self.ptr)
            self.ptr = 0

    cpdef load_file(self, str filename):
        runtime._ensure_context()
        self.ptr = driver.moduleLoad(filename)

    cpdef load(self, bytes cubin):
        runtime._ensure_context()
        self.ptr = driver.moduleLoadData(cubin)

    cpdef get_global_var(self, str name):
        return driver.moduleGetGlobal(self.ptr, name)

    cpdef get_function(self, str name):
        return Function(self, name)


cdef class LinkState:

    """CUDA link state."""

    def __init__(self):
        self.ptr = driver.linkCreate()

    def __del__(self):
        if self.ptr:
            driver.linkDestroy(self.ptr)
            self.ptr = 0

    cpdef add_ptr_data(self, unicode data, unicode name):
        cdef bytes data_byte = data.encode()
        driver.linkAddData(self.ptr, driver.CU_JIT_INPUT_PTX, data_byte, name)

    cpdef bytes complete(self):
        cubin = driver.linkComplete(self.ptr)
        return cubin
