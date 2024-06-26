import glob
import os
import setuptools
import sys
import torch.utils.cpp_extension


# This is needed for versioneer to be importable when building with PEP 517.
# See <https://github.com/warner/python-versioneer/issues/193> and links
# therein for more information.
sys.path.append(os.path.dirname(__file__))
import versioneer


EXTENSIONS = []
CMD_CLASS = {}


def add_cpp_extension():
    extra_compile_args = [
        '-std=c++17' if not sys.platform.startswith('win') else '/std:c++17',
    ]
    extra_link_args = []
    define_macros = [
        ('_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS', None),  # mostly for the pytorch codebase
    ]

    if sys.platform.startswith('win'):
        extra_compile_args += ['/permissive']
        define_macros += [('OPENPIFPAF_DLLEXPORT', None)]

    if os.getenv('DEBUG', '0') == '1':
        print('DEBUG mode')
        if sys.platform.startswith('linux'):
            extra_compile_args += ['-g', '-Og']
            extra_compile_args += [
                '-Wuninitialized',
                # '-Werror',  # fails in pytorch code, but would be nice to have in CI
            ]
        define_macros += [('DEBUG', None)]

    this_dir = os.path.dirname(os.path.abspath(__file__))
    csrc_dir = os.path.join(this_dir, 'src', 'openpifpaf', 'csrc')
    EXTENSIONS.append(
        torch.utils.cpp_extension.CppExtension(
            'openpifpaf._cpp',
            glob.glob(os.path.join(csrc_dir, 'src', '**', '*.cpp'), recursive=True),
            depends=glob.glob(os.path.join(csrc_dir, 'include', '**', '*.hpp'), recursive=True),
            include_dirs=[os.path.join(csrc_dir, 'include')],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )
    assert 'build_ext' not in CMD_CLASS, f"build_ext in CMD_CLASS: {CMD_CLASS}"
    CMD_CLASS['build_ext'] = torch.utils.cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)


add_cpp_extension()
CMD_CLASS = versioneer.get_cmdclass(CMD_CLASS)
setuptools.setup(
    name='openpifpaf-vita',
    version=versioneer.get_version(),
    license='GNU AGPLv3',
    description='PifPaf: Composite Fields for Pose Estimation',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='EPFL VITA',
    author_email='epfl.vita@gmail.com',
    url='https://github.com/vita-epfl/openpifpaf',

    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    package_data={
        'openpifpaf': ['*.dll', '*.dylib', '*.so'],
    },
    cmdclass=CMD_CLASS,
    ext_modules=EXTENSIONS,
    zip_safe=False,

    python_requires='>=3.7',
    install_requires=[
        'importlib_metadata!=3.8.0',  # temporary for pytest
        'numpy>=1.16',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'pillow!=8.3.0',  # exclusion torchvision 0.10.0 compatibility
    ],
    extras_require={
        'backbones': [
            'timm>=0.4.9',  # For Swin Transformer and XCiT
            'einops>=0.3',  # required for BotNet
            'mmengine',
            'mmcv>=2.0',
            'mmpose>=1.0',  # for HRFormer
            'mmpretrain',  # for ConvNeXt V2
            'open_clip_torch',  # for CLIP-ConvNeXt
        ],
        'dev': [
            'flameprof',
            'ipython<8',  # temporarily added to avoid broken output cells in jupyter-book
            'jsonschema[format-nongpl]==4.17.3',
            'jupyter-book>=0.9.1,<0.14',  # breaking changes introduced in 0.14
            'matplotlib>=3.3',
            'nbdime',
            'nbstripout<0.7',  # breaking changes introduced in 0.7
            'scipy',
            'sphinx-book-theme',
            'sphinxcontrib-applehelp<=1.0.4',
            'sphinxcontrib-devhelp<=1.0.2',
            'sphinxcontrib-htmlhelp<=2.0.1',
            'sphinxcontrib-jsmath<=1.0.1',
            'sphinxcontrib-qthelp<=1.0.3',
            'sphinxcontrib-serializinghtml<=1.1.5',
            'wheel',
        ],
        'onnx': [
            'onnx',
            'onnxruntime; python_version<="3.9"',
            'protobuf<4',  # temporary explicit dependency until tests pass again
        ],
        'coreml': [
            'coremltools>=5.0b3,<7',
        ],
        'test': [
            'cpplint',
            'nbconvert<7',
            'nbstripout<0.7',  # breaking changes introduced in 0.7
            'nbval',
            'opencv-python',
            'pycodestyle',
            'pylint<2.9.4',  # avoid 2.9.4 and up for time.perf_counter deprecation warnings
            'pytest',
            'requests>=2.6.0',
            'tabulate',
            'thop',
        ],
        'train': [
            'matplotlib>=3.3',  # required by pycocotools
            'pycocotools@git+https://github.com/ppwwyyxx/cocoapi.git#subdirectory=PythonAPI',  # pre-install cython, last version fixing compilation not on pypi
            'scipy',
            'xtcocotools>=1.5; sys_platform == "linux"',  # pre-install numpy. required for wholebody eval, only wheels and only for linux on pypi
        ],
    },
)
