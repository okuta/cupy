from libcpp cimport vector
from libcpp cimport map

from cupy.cuda cimport device


cdef class MemoryPointer:

    cdef:
        readonly device.Device device
        readonly object mem
        readonly size_t ptr

    cpdef copy_from_device(self, MemoryPointer src, Py_ssize_t size)
    cpdef copy_from_device_async(self, MemoryPointer src, size_t size,
                                 stream=?)
    cpdef copy_from_host(self, mem, size_t size)
    cpdef copy_from_host_async(self, mem, size_t size, stream=?)
    cpdef copy_from(self, mem, size_t size)
    cpdef copy_from_async(self, mem, size_t size, stream=?)
    cpdef copy_to_host(self, mem, size_t size)
    cpdef copy_to_host_async(self, mem, size_t size, stream=?)
    cpdef memset(self, int value, size_t size)
    cpdef memset_async(self, int value, size_t size, stream=?)


cpdef MemoryPointer alloc(Py_ssize_t size)


cpdef set_allocator(allocator=*)


cdef class SingleDeviceMemoryPool:

    cdef:
        object _allocator
        dict _in_use
        dict _arena_pool
        object __weakref__
        object _weakref
        object _free_lock
        object _in_use_lock
        readonly int _device_id

    cdef _arena(self, size_t stream_ptr)
    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef MemoryPointer _malloc(self, Py_ssize_t size)
    cdef _free(self, size_t ptr, Py_ssize_t size, size_t stream_ptr)
    cdef _free_all_blocks(self, size_t stream_ptr)
    cpdef free_all_blocks(self, stream=?, stream_ptr=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)

cdef class MemoryPool:

    cdef:
        object _pools

    cpdef MemoryPointer malloc(self, Py_ssize_t size)
    cpdef free_all_blocks(self, stream=?, stream_ptr=?)
    cpdef free_all_free(self)
    cpdef n_free_blocks(self)
    cpdef used_bytes(self)
    cpdef free_bytes(self)
    cpdef total_bytes(self)
