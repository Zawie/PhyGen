# PhyGen
Just a system to generate sequences to run the model on

# Installation
Listed below are the packages and commands you need to get started!

## PyTorch
Go to PyTorch.org for information on how to download for your system. **WARNING** Make sure you have 64 bit python installed, otherwise the pip install for PyTorch will not work.

For the latest stable release as of June 5, 2020:
### Windows, 10.2 CUDA
```
pip install torch===1.5.0 torchvision===0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### Mac, no cuda
```
pip install torch torchvision
```
for CUDA, download from the source code at PyTorch's github page.


## Pyvolve
PyVolve is a library that we use to generate genetic sequences from phylogenetic trees. 
For documentation, follow this link: 
https://github.com/sjspielman/pyvolve/blob/master/user_manual/pyvolve_manual.pdf

To install pyvolve, run the command 
```
pip install pyvolve
```
in the command prompt.

## Pickle

Run the command 
```
pip install pickle
```
in the command prompt.
Be aware that, depending on your version of python, that the default protocol for pickle is different and may cause errors.
For python 3.8.3, default is 4. Older versions are 3 or even 2 if you're using python2

