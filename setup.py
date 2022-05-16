from distutils.command.install import install

from setuptools import setup

r_packages = {
    "LTRCtrees": "https://cran.r-project.org/src/contrib/Archive/LTRCtrees/LTRCtrees_1.1.0.tar.gz",
    "Icens": "https://cran.r-project.org/src/contrib/Archive/Icens/Icens_1.24.0.tar.gz",
    "interval": "https://cran.r-project.org/src/contrib/interval_1.1-0.8.tar.gz"
    }

r_packages_win = {
    "LTRCtrees": "https://cran.microsoft.com/snapshot/2017-08-01/bin/windows/contrib/3.4/LTRCtrees_0.5.0.zip",
    "Icens": "https://cran.microsoft.com/snapshot/2017-08-01/bin/windows/contrib/3.4/Icens_1.24.0.tar.gz",
    "interval": "https://cran.r-project.org/bin/windows/contrib/4.2/interval_1.1-0.8.zip"
}


class PostInstall(install):
    def run(self):
        import sys
        import rpy2.robjects as ro
        r_session = ro.r
        install.run(self)
        package_list = ["partykit", "inum", "icenReg", 
                        "survival", "data.table",
                        "Formula", "rpart", 'stringi', "hash"]
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
            packages_to_compile = r_packages
        else:
            packages_to_compile = r_packages_win
        for k, v in packages_to_compile.items():
            r_cmd2 = """if (!require('""" + k + """', character.only = TRUE)) {
                      install.packages('""" + v + """')
                    }"""
            r_session(r_cmd2)


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    line_iter = (line.strip() for line in open(filename))
    return [line for line in line_iter if line and not line.startswith("#")]


install_req = parse_requirements("requirements.txt")

setup(name='survival_trees',
      version='0.0.10',
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
