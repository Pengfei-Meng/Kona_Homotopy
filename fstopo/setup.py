from numpy.distutils.misc_util import Configuration

config = Configuration(
    package_name='fstopo',
    package_path='src'
)

config.add_subpackage(
    subpackage_name='problems',
    subpackage_path='src/problems'
)

config.add_extension(
    'multi',
    ['src/fortran/precision.f90',
     'src/fortran/element.f90',
     'src/fortran/multi.f90']
)

config.add_extension(
    'thickness',
    ['src/fortran/precision.f90',
     'src/fortran/element.f90',
     'src/fortran/thickness.f90']
)

config.add_extension(
    'kona',
    ['src/fortran/precision.f90',
     'src/fortran/element.f90',
     'src/fortran/kona.f90']
)

config.add_extension(
    'sparse',
    ['src/fortran/quicksort.f90',
     'src/fortran/precision.f90',
     'src/fortran/sparse.f90']
)

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**config.todict())
