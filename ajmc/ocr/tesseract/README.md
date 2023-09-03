Here is a brief vademecum on how to install tesseract with training tools on a linux machine without sudo. 

**Note**. Training on mac did not work properly despite a meticulous installation. 


# Linux install with training tools

1) Install Conda in your local dir
2) Install all dependencies with conda, so that bin, lib and include are respectively in ``$CONDA_DIR``
	- Cairo, pango, icu, leptonica

3) Git clone tesseract
4) Run autogen:
```
cd tesseract
.autogen.sh
```
5) Then configure
```
./configure \  
               --prefix=/scratch/sven/anaconda3/ \  
               --with-extra-libraries=/scratch/sven/anaconda3/lib \  
               --with-extra-includes=/scratch/sven/anaconda3/include \  
               --with-curl=no \  
               PKG_CONFIG_PATH="/scratch/sven/anaconda3/lib/pkgconfig"  
               CXXFLAGS="-I/scratch/sven/ocr_exp/lib/local/include" \  
               LDFLAGS="-L/scratch/sven/ocr_exp/lib/local/lib" \  
               CFLAGS="-I/scratch/sven/ocr_exp/lib/local/include" \  
               pango_CFLAGS="-I/scratch/sven/anaconda3/include/pango" \  
               pango_LIBS="-L/scratch/sven/anaconda3/lib" \  
               cairo_CFLAGS="-I/scratch/sven/anaconda3/include/pango" \  
               cairo_LIBS="-L/scratch/sven/anaconda3/lib -lcairo" \

```

6) You should see something like: (and run the following (without sudo ldconfig))
```
#Configuration is done.  
#You can now build and install tesseract by running:  

make  
make install  

#Documentation will not be built because asciidoc or xsltproc is missing.  

#Training tools can be built and installed with:  
# 

make training  
make training-install
```


## Problems

```bash
# if ... version required by bash
conda install -c conda-forge ncurses
```

If problems with liblept, ⚠️ add this to configure : 
```bash
LIBLEPT_HEADERSDIR=$CONDA_PREFIX/lib 
./configure \
               --prefix=$CONDA_PREFIX/ \
               --with-extra-libraries=$CONDA_PREFIX/lib
```

then add this: 
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LD_LIBRARY_PATH/lib/:$CONDA_PREFIX/lib/
```

## Tesstrain details

1) Git clone tesstrain

2) Follow the training instructions in tesstrain's [readme](https://github.com/tesseract-ocr/tesstrain).
3) Your command should look like :
```shell

```