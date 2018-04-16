import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

import six

from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import nvrtc
from cupy.cuda import runtime

_nvrtc_version = None
_nvrtc_max_compute_capability = None


def _get_nvrtc_version():
    global _nvrtc_version
    if _nvrtc_version is None:
        _nvrtc_version = nvrtc.getVersion()

    return _nvrtc_version


def _get_arch():
    global _nvrtc_max_compute_capability
    if _nvrtc_max_compute_capability is None:
        # See Supported Compile Options section of NVRTC User Guide for
        # the maximum value allowed for `--gpu-architecture`.
        major, minor = _get_nvrtc_version()
        if major < 9:
            # CUDA 7.0 / 7.5 / 8.0
            _nvrtc_max_compute_capability = '53'
        else:
            # CUDA 9.0 / 9.1
            _nvrtc_max_compute_capability = '72'
    cc = min(device.Device().compute_capability, _nvrtc_max_compute_capability)
    return 'compute_%s' % cc


class TemporaryDirectory(object):
    def __enter__(self):
        self.path = tempfile.mkdtemp()
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            return

        for name in os.listdir(self.path):
            os.unlink(os.path.join(self.path, name))
        os.rmdir(self.path)


def _get_bool_env_variable(name, default):
    val = os.environ.get(name)
    if val is None or len(val) == 0:
        return default
    try:
        return int(val) == 1
    except ValueError:
        return False


def compile_using_nvrtc(source, options=(), arch=None):
    if not arch:
        arch = _get_arch()

    options += ('-arch={}'.format(arch),)

    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cu' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        prog = _NVRTCProgram(source, cu_path)
        try:
            ptx = prog.compile(options)
        except CompileException as e:
            dump = _get_bool_env_variable(
                'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
            if dump:
                e.dump(sys.stderr)
            raise

        return ptx


def _preprocess(source, options, arch):
    options += ('-arch={}'.format(arch),)

    prog = _NVRTCProgram(source, '')
    try:
        result = prog.compile(options)
    except CompileException as e:
        dump = _get_bool_env_variable(
            'CUPY_DUMP_CUDA_SOURCE_ON_ERROR', False)
        if dump:
            e.dump(sys.stderr)
        raise

    assert isinstance(result, six.text_type)
    return result


_default_cache_dir = os.path.expanduser('~/.cupy/kernel_cache')


def get_cache_dir():
    return os.environ.get('CUPY_CACHE_DIR', _default_cache_dir)


_empty_file_preprocess_cache = {}


def compile_with_cache(source, options=(), arch=None, cache_dir=None,
                       extra_source=None):
    if runtime.is_hip:
        return _compile_with_cache_hipcc(source, options, arch, cache_dir,
                                         extra_source)
    else:
        return _compile_with_cache_nvrtc(source, options, arch, cache_dir,
                                         extra_source)


def _compile_with_cache_nvrtc(source, options, arch, cache_dir,
                              extra_source):
    # NVRTC does not use extra_source. extra_source is used for cache key.
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = _get_arch()

    options += ('-ftz=true',)

    env = (arch, options, _get_nvrtc_version())
    base = _empty_file_preprocess_cache.get(env, None)
    if base is None:
        # This is checking of NVRTC compiler internal version
        base = _preprocess('', options, arch)
        _empty_file_preprocess_cache[env] = base
    key_src = '%s %s %s %s' % (env, base, source, extra_source)

    key_src = key_src.encode('utf-8')
    name = '%s_2.cubin' % hashlib.md5(key_src).hexdigest()

    if not os.path.isdir(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError:
            if not os.path.isdir(cache_dir):
                raise

    mod = function.Module()
    # To handle conflicts in concurrent situation, we adopt lock-free method
    # to avoid performance degradation.
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = file.read()
        if len(data) >= 32:
            hash = data[:32]
            cubin = data[32:]
            cubin_hash = six.b(hashlib.md5(cubin).hexdigest())
            if hash == cubin_hash:
                mod.load(cubin)
                return mod

    ptx = compile_using_nvrtc(source, options, arch)
    ls = function.LinkState()
    ls.add_ptr_data(ptx, six.u('cupy.ptx'))
    cubin = ls.complete()
    cubin_hash = six.b(hashlib.md5(cubin).hexdigest())

    # shutil.move is not atomic operation, so it could result in a corrupted
    # file. We detect it by appending md5 hash at the beginning of each cache
    # file. If the file is corrupted, it will be ignored next time it is read.
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
        tf.write(cubin_hash)
        tf.write(cubin)
        temp_path = tf.name
    shutil.move(temp_path, path)

    # Save .cu source file along with .cubin
    if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
        with open(path + '.cu', 'w') as f:
            f.write(source)

    mod.load(cubin)
    return mod


class CompileException(Exception):

    def __init__(self, msg, source, name, options):
        self._msg = msg
        self.source = source
        self.name = name
        self.options = options

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.get_message()

    def get_message(self):
        return self._msg

    def dump(self, f):
        lines = self.source.split('\n')
        digits = int(math.floor(math.log10(len(lines)))) + 1
        linum_fmt = '{{:0{}d}} '.format(digits)
        f.write('NVRTC compilation error: {}\n'.format(self))
        f.write('-----\n')
        f.write('Name: {}\n'.format(self.name))
        f.write('Options: {}\n'.format(' '.join(self.options)))
        f.write('CUDA source:\n')
        for i, line in enumerate(lines):
            f.write(linum_fmt.format(i + 1) + line.rstrip() + '\n')
        f.write('-----\n')
        f.flush()


class _NVRTCProgram(object):

    def __init__(self, src, name="default_program", headers=(),
                 include_names=()):
        self.ptr = None

        if isinstance(src, six.binary_type):
            src = src.decode('UTF-8')
        if isinstance(name, six.binary_type):
            name = name.decode('UTF-8')

        self.src = src
        self.name = name
        self.ptr = nvrtc.createProgram(src, name, headers, include_names)

    def __del__(self):
        if self.ptr:
            nvrtc.destroyProgram(self.ptr)

    def compile(self, options=()):
        try:
            nvrtc.compileProgram(self.ptr, options)
            return nvrtc.getPTX(self.ptr)
        except nvrtc.NVRTCError:
            log = nvrtc.getProgramLog(self.ptr)
            raise CompileException(log, self.src, self.name, options)


def is_valid_kernel_name(name):
    return re.match('^[a-zA-Z_][a-zA-Z_0-9]*$', name) is not None


_hipcc_version = None


def _get_hipcc_version():
    global _hipcc_version
    if _hipcc_version is None:
        cmd = ['hipcc', '--version']
        _hipcc_version = _run_hipcc(cmd)
    return _hipcc_version


def _run_hipcc(cmd, cwd='.', env=None):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT,
                                       cwd=cwd, env=env)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            '`hipcc` command returns non-zero exit status. \n'
            'command: {0}\n'
            'return-code: {1}\n'
            'stdout/stderr: \n'
            '{2}'.format(e.cmd, e.returncode, e.output))
    except OSError as e:
        raise OSError('Failed to run `hipcc` command. '
                      'Check PATH environment variable: '
                      + str(e))


def _run_hipcc_convert(in_path, out_path, cwd):
    cmd = '''
kernels=$(objdump -t "{input}" | grep grid_launch_parm \
    | sed 's/ \+/ /g; s/\t/ /g' | cut -d" " -f6)
map_sym=""
for mangled_sym in $kernels; do
  real_sym=$(c++filt $(c++filt $mangled_sym|cut -d: -f3|sed 's/_functor//g') \
      | cut -d\( -f1)
  map_sym="--redefine-sym $mangled_sym=$real_sym $map_sym"
done
objcopy -F elf64-little $map_sym "{input}" "{output}"
'''.format(input=in_path, output=out_path)
    try:
        return subprocess.check_output(cmd, cwd=cwd, shell=True,
                                       stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            'covert command returns non-zero exit status. \n'
            'command: {0}\n'
            'return-code: {1}\n'
            'stdout/stderr: \n'
            '{2}'.format(e.cmd, e.returncode, e.output))
    except OSError as e:
        raise OSError('Failed to run convert command. '
                      'Check PATH environment variable: '
                      + str(e))


def _hipcc(source, options, arch):
    cmd = ['hipcc', '-DGENERIC_GRID_LAUNCH=0'] + list(options)

    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        in_path = path + '.cpp'
        tmp_path = os.path.join(root_dir, 'dump-%s.hsaco' % arch)
        dummy_out_path = path + '.out'
        out_path = path + '.hasco'

        with open(in_path, 'w') as f:
            f.write(source)

        cmd += [in_path, '-o', dummy_out_path]
        env = os.environ.copy()
        env['KMDUMPISA'] = '1'

        output = _run_hipcc(cmd, root_dir, env)
        if not os.path.isfile(tmp_path):
            raise RuntimeError(
                '`hipcc` command does not generate output file. \n'
                'command: {0}\n'
                'stdout/stderr: \n'
                '{1}'.format(cmd, output))
        _run_hipcc_convert(tmp_path, out_path, root_dir)
        with open(out_path, 'rb') as f:
            return f.read()


def _preprocess_hipcc(source, options):
    cmd = ['hipcc', '--preprocess'] + list(options)
    with TemporaryDirectory() as root_dir:
        path = os.path.join(root_dir, 'kern')
        cu_path = '%s.cpp' % path

        with open(cu_path, 'w') as cu_file:
            cu_file.write(source)

        cmd.append(cu_path)
        pp_src = _run_hipcc(cmd, root_dir)
        assert isinstance(pp_src, six.binary_type)
        return re.sub(b'(?m)^#.*$', b'', pp_src)


def _convert_to_hip_source(source):
    table = [
        ('extern "C"', ''),
        ('threadIdx.', 'hipThreadIdx_'),
        ('blockIdx.', 'hipBlockIdx_'),
        ('blockDim.', 'hipBlockDim_'),
        ('gridDim.', 'hipGridDim_'),
    ]
    for i, j in table:
        source = source.replace(i, j)
    return source


def _compile_with_cache_hipcc(source, options, arch, cache_dir, extra_source,
                              use_converter=True):
    global _empty_file_preprocess_cache
    if cache_dir is None:
        cache_dir = get_cache_dir()
    if arch is None:
        arch = os.environ.get('HCC_AMDGPU_TARGET')
    if use_converter:
        source = _convert_to_hip_source(source)

    env = (arch, options, _get_hipcc_version())
    key_src = '%s %s %s' % (
        env, _preprocess_hipcc(source, options), extra_source)

    key_src = key_src.encode('utf-8')
    name = '%s_2.hsaco' % hashlib.md5(key_src).hexdigest()

    if not os.path.isdir(cache_dir):
        try:
            os.makedirs(cache_dir)
        except OSError:
            if not os.path.isdir(cache_dir):
                raise

    mod = function.Module()
    # To handle conflicts in concurrent situation, we adopt lock-free method
    # to avoid performance degradation.
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            data = file.read()
        if len(data) >= 32:
            hash_value = data[:32]
            binary = data[32:]
            binary_hash = six.b(hashlib.md5(binary).hexdigest())
            if hash_value == binary_hash:
                mod.load(binary)
                return mod

    binary = _hipcc(source, options, arch)
    binary_hash = six.b(hashlib.md5(binary).hexdigest())

    # shutil.move is not atomic operation, so it could result in a corrupted
    # file. We detect it by appending md5 hash at the beginning of each cache
    # file. If the file is corrupted, it will be ignored next time it is read.
    with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tf:
        tf.write(binary_hash)
        tf.write(binary)
        temp_path = tf.name
    shutil.move(temp_path, path)

    # Save .cu source file along with .hsaco
    if _get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', False):
        with open(path + '.cu', 'w') as f:
            f.write(source)
    print(path)

    mod.load(binary)
    return mod
