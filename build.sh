# Build R 3.5
add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
# apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
gpg -a --export E298A3A825C0D65DFD57CBB651716619E084DAB9 | apt-key add -
install r-base r-base-core r-recommended

# Build up R dependancies
pip install rpy2

Rscript -e 'install.packages("partykit")'
Rscript -e 'install.packages("inum")'
Rscript -e 'install.packages("icenReg")'
Rscript -e 'install.packages("survival")'
Rscript -e 'install.packages("data.table")'
Rscript -e 'install.packages("Formula")'
Rscript -e 'install.packages("rpart")'
Rscript -e 'install.packages("stringi")'
Rscript -e 'install.packages("hash")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/LTRCtrees/LTRCtrees_1.1.0.tar.gz")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/Archive/Icens/Icens_1.24.0.tar.gz")'
Rscript -e 'install.packages("https://cran.r-project.org/src/contrib/interval_1.1-0.8.tar.gz")'


# Finally install algorithm
pip install git+https://github.com/eurobios-scb/survival-trees.git
