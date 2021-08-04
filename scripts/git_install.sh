#!/bin/bash
#fail if any errors
wget -O git.tar.gz https://github.com/git/git/archive/v2.17.0.tar.gz
tar zxf git.tar.gz
mv git-2.17.0 git
cd git
make configure
./configure --prefix=`pwd` --with-curl --with-expat
# ./configure --prefix=`pwd`
# Make flags from https://public-inbox.org/git/CAP8UFD2gKTourXUdB_9_FZ3AEECTDc1Fx1NFKzeaTZDWHC3jxA@mail.gmail.com/
make NO_GETTEXT=Nope NO_TCLTK=Nope
make install NO_GETTEXT=Nope NO_TCLTK=Nope

