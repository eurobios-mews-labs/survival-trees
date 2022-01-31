from setuptools import setup


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    line_iter = (line.strip() for line in open(filename))
    return [line for line in line_iter if line and not line.startswith("#")]


install_req = parse_requirements("requirements.txt")

setup(name='survival_trees',
      version='0.0.6',
      description='Python survival trees',
      url='https://gitlab.eurobios.com/vlaurent/survival-trees',
      author='Vincent LAURENT',
      author_email='vlaurent@eurobios.com',
      license='BSD',
      install_requires=install_req,
      packages=['survival_trees', 'survival_trees.tools'],
      package_dir={'pypgsql': 'survival_trees',
                   'survival_trees.tools': 'survival_trees/tools',
                   },
      include_package_data=True,
      package_data={'survival': ['*.R']},
      zip_safe=False)
