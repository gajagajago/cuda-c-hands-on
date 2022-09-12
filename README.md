# CUDA C Practice

Hands-on exercises from 'Programming Massively Parallel Processors' 3rd Edition (Kirk & Hwu)

<image src="./etc/cover.jpg" width=200px height=250px></image>

## How to build
1. Install CUDA
* Install NVIDIA's official release from [link](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=18.04&target_type=runfile_local)
  * Under your Linux distro, select the runfile(local) option. On a server, one easy way is to copy the <LINK> of the Download button and, in any location of your home directory, run wget <LINK>. It will download the <INSTALLER>file.
  * Execute the <INSTALLER> file (run this command:  sh <INSTALLER>)
  * Accept the EULA.
  * Deselect driver installation (pressing ENTER on it).
  * Go to Options -> Toolkit Options. Select Change Toolkit Install Path. Change it to a non-sudo directory.
  * Deselect Options -> Toolkit Options -> Create symbolic link from /usr/local/cuda.
  * Deselect Options -> Toolkit Options -> Create desktop menu shortcuts.
  * Deselect Options -> Toolkit Options -> Install manpage documents to /usr/share/man.
  * Go to Options. Select Library install path (Blank for system default). Use the same path for “Library install path” and “Toolkit Install Path”. 
* Toolkit/Library path for me: `/home/gajagajago/cuda-11.7/`
* Export paths to your terminal's configuration file. (.zshrc OR .bashrc)
``` zsh
export PATH=/home/gajagajago/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/home/gajagajago/cuda-11.7/lib64:$LD_LIBRARY_PATH
``` 
* Apply the edited path
``` zsh
source ~/.zshrc # OR ~/.bashrc
```