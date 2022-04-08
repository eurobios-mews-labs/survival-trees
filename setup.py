from distutils.command.install import install

from setuptools import setup


class PostInstall(install):
    def run(self):
        import sys
        import rpy2.robjects as ro
        r_session = ro.r
        install.run(self)
        package_list = ["survival", "LTRCtrees", "data.table",
         "rpart", "Icens", "interval", 'stringi', "hash"]
        if len(package_list) == 1:
            package_list = f"('{package_list[0]}')"
        else:
            package_list = str(tuple(
                package_list
            ))
        r_cmd = """
                   packages = c""" + package_list + """
                   package.check <- lapply(
                     packages,
                     FUN = function(x) {
                       if (!require(x, character.only = TRUE)) {
                         install.packages(x, dependencies = TRUE)
                         library(x, character.only = TRUE)
                       }
                     }
                   )"""
        r_session(r_cmd)

        if "linux" in sys.platform:
            r_cmd2 = """if (!require("LTRCtrees", character.only = TRUE)) {
                      install.packages("https://cran.r-project.org/src/contrib/Archive/LTRCtrees/LTRCtrees_1.1.0.tar.gz")
                    }"""
        else:
            r_cmd2 = """if (!require("LTRCtrees", character.only = TRUE)) {
                      install.packages("https://cran.microsoft.com/snapshot/2017-08-01/bin/windows/contrib/3.4/LTRCtrees_0.5.0.zip")
                    }"""

        r_session(r_cmd2)
        # raise AssertionError("Passed")


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    line_iter = (line.strip() for line in open(filename))
    return [line for line in line_iter if line and not line.startswith("#")]


install_req = parse_requirements("requirements.txt")

setup(name='survival_trees',
      version='0.0.7',
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
      package_data={'survival_trees': ['*.R']},
      cmdclass=dict(install=PostInstall),
      zip_safe=False)
