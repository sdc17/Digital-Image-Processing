## requirement:
1. conda(miniconda or anaconda)
2. vscode(ide) download [here](https://code.visualstudio.com)
   1. jupyter extension
   2. python extension
3. python env:
    1. numpy
    2. jupyter

## Windows user

watch this -> https://www.youtube.com/watch?v=3Wt00qGlh3s
download here: https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda
## Linux/MacOS

```bash
mkdir ~/envs
cd envs
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Min* -b -p ~/envs/miniconda3
```

```bash
conda create -n DIVP21fall jupyter numpy scipy imageio matplotlib
```