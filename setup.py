from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.ccompiler import CCompiler
from distutils.errors import DistutilsExecError, CompileError
from distutils import log
import os, glob, numpy, platform, sys,subprocess
if 'upload' in sys.argv or 'register' in sys.argv:
    from ez_setup import use_setuptools; use_setuptools()
    from setuptools import setup, Extension

print "Generating src/__version__.py: ",
__version__ = open('VERSION').read().strip()
print __version__
open('src/__version__.py','w').write('__version__="%s"'%__version__)

#read the latest git status out to an installed file
try:
#    gitbranch = subprocess.check_output('git symbolic-ref -q HEAD',shell=True, cwd='.').strip().split('/')[-1]
    gitbranch = os.popen('git symbolic-ref -q HEAD').read().strip()
    print "Generating src/__branch__.py"
#    gitlog = subprocess.check_output('git log -n1 --pretty="%h%n%s%n--%n%an%n%ae%n%ai"',shell=True, cwd='.').strip()
    gitlog = os.popen('git log -n1 --pretty="%h%n%s%n--%n%an%n%ae%n%ai"').read().strip()
    print "Generating src/__gitlog__.py."
    print gitlog
except:
    gitbranch = "unknown branch"
    gitlog = "git log not found"
open('src/__branch__.py','w').write('__branch__ = \"%s\"'%gitbranch)
open('src/__gitlog__.py','w').write('__gitlog__ = \"\"\"%s\"\"\"'%gitlog)


def get_description():
    lines = [L.strip() for L in open('README').readlines()]
    d_start = None
    for cnt, L in enumerate(lines):
        if L.startswith('DESCRIPTION'): d_start = cnt + 1
        elif not d_start is None:
            if len(L) == 0: return ' '.join(lines[d_start:cnt])
    raise RuntimeError('Bad README')

def indir(path, files):
    return [os.path.join(path, f) for f in files]

NVCC, CUDA_DIR = '', ''
for path in ('/usr/local/cuda', '/opt/cuda', '/usr/local/cuda-5.0'):
    if os.path.exists(path):
        CUDA_DIR = path
        NVCC = os.path.join(path, 'bin', 'nvcc')
        break
if CUDA_DIR == '': log.warn("No CUDA installation was found.  Trying anyway...")

class CudaCompiler(CCompiler):
    compiler_type = 'nvcc'
    compiler_so = [NVCC]
    executables = {'compiler' : [NVCC]}
    src_extensions = ['.cu']
    obj_extension = '.o'
    def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
        try: self.spawn(self.compiler_so + cc_args + [src,'-o',obj] + extra_postargs)
        except DistutilsExecError, msg: raise CompileError, msg

class CudaExtension(Extension):
    def __init__(self, name, sources, **kwargs):
        Extension.__init__(self, name, sources, **kwargs)
        is64 = platform.architecture()[0].startswith('64')
        self.libraries.append('cudart')
        self.libraries.append('cufft')
        self.include_dirs.append(os.path.join(CUDA_DIR, 'include'))
        if is64 and os.path.exists(os.path.join(CUDA_DIR,'lib64')):
            self.library_dirs.append(os.path.join(CUDA_DIR,'lib64'))
        else: self.library_dirs.append(os.path.join(CUDA_DIR, 'lib'))
        try: self.cuda_sources = kwargs.pop('cuda_sources')
        except(KeyError): self.cuda_sources = []
        try: self.cuda_extra_compile_args = kwargs.pop('cuda_extra_compile_args')
        except(KeyError): self.cuda_extra_compile_args = []
        # NVCC wants us to call out 64/32-bit compiling explicitly
        if is64: self.cuda_extra_compile_args.append('-m64')
        else: self.cuda_extra_compile_args.append('-m32')
        self.cuda_extra_compile_args.append('-Xcompiler')
        self.cuda_extra_compile_args.append('-fPIC')
        #self.cuda_extra_compile_args.append('-g')
        #self.cuda_extra_compile_args.append('-G')
        #self.cuda_extra_compile_args.append('-arch=sm_35')
        self.cuda_extra_compile_args.append('-arch=sm_30')
        
class cuda_build_ext(build_ext):
    def build_extension(self, ext):
        if isinstance(ext, CudaExtension):
            log.info("pre-building '%s' CudaExtension using nvcc", ext.name)
            compiler = CudaCompiler()
            objects = compiler.compile(ext.cuda_sources, output_dir=self.build_temp,
                extra_postargs=ext.cuda_extra_compile_args)
            ext.extra_objects += objects
        build_ext.build_extension(self, ext)

setup(name = 'aipy',
    version = __version__,
    description = 'Astronomical Interferometry in PYthon',
    long_description = get_description(),
    license = 'GPL',
    author = 'Aaron Parsons',
    author_email = 'aparsons@astron.berkeley.edu',
    url = 'http://setiathome.berkeley.edu/~aparsons/aipy/aipy.cgi',
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    setup_requires = ['numpy>=1.2'],
    install_requires = ['pyephem>=3.7.3.2', 'pyfits>=2.1', 'numpy>=1.2'],
    dependency_links = [
        'http://www.stsci.edu/resources/software_hardware/pyfits'
    ],
    package_dir = {'aipy':'src', 'aipy.optimize':'src/optimize', 'aipy._src':'src/_src'},
    packages = ['aipy', 'aipy.optimize','aipy._src'],
    ext_modules = [
        Extension('aipy._healpix',
            ['src/_healpix/healpix_wrap.cpp', 
            'src/_healpix/cxx/Healpix_cxx/healpix_base.cc'],
            include_dirs = [numpy.get_include(), 'src/_healpix/cxx/cxxsupport',
                'src/_healpix/cxx/Healpix_cxx']),
        Extension('aipy._alm',
            ['src/_healpix/alm_wrap.cpp', 
            'src/_healpix/cxx/Healpix_cxx/alm_map_tools.cc',
            'src/_healpix/cxx/libfftpack/ls_fft.c',
            'src/_healpix/cxx/libfftpack/bluestein.c',
            'src/_healpix/cxx/libfftpack/fftpack.c',
            'src/_healpix/cxx/Healpix_cxx/healpix_map.cc',
            'src/_healpix/cxx/Healpix_cxx/healpix_base.cc'],
            include_dirs = [numpy.get_include(), 'src/_healpix/cxx/cxxsupport',
                'src/_healpix/cxx/Healpix_cxx']),
        Extension('aipy._miriad', ['src/_miriad/miriad_wrap.cpp'] + \
            indir('src/_miriad/mir', ['uvio.c','hio.c','pack.c','bug.c',
                'dio.c','headio.c','maskio.c']),
            include_dirs = [numpy.get_include(), 'src/_miriad', 
                'src/_miriad/mir']),
        Extension('aipy._deconv', ['src/_deconv/deconv.cpp'],
            include_dirs = [numpy.get_include()]),
        CudaExtension('aipy._deconvGPU', ['src/_deconvGPU/wrap_deconv.cpp'],
            cuda_sources = ['src/_deconvGPU/deconv.cu'],
            include_dirs = ['src/_deconvGPU', numpy.get_include()],),
        #Extension('aipy._img', ['src/_img/img.cpp'],
        #    include_dirs = [numpy.get_include()]),
        Extension('aipy._dsp', ['src/_dsp/dsp.c', 'src/_dsp/grid/grid.c'],
            include_dirs = [numpy.get_include(), 'src/_dsp', 'src/_dsp/grid']),
        Extension('aipy.utils', ['src/utils/utils.cpp'],
            include_dirs = [numpy.get_include()]),
        Extension('aipy._cephes',
            ['src/_cephes/_cephesmodule.c', 'src/_cephes/ufunc_extras.c'] + \
            glob.glob('src/_cephes/cephes/*.c') + \
            glob.glob('src/_cephes/c_misc/*.c'),
            include_dirs = [numpy.get_include()]),
    ],
    scripts=glob.glob('scripts/*'),
    package_data = {'aipy': ['_src/*.txt']},
    cmdclass = {'build_ext': cuda_build_ext},
)
