Need to install CUDA Toolkit (not just drivers!) 12.5 or 12.8 (12.6 will not work)

To build this repo:

```
pip install --upgrade "jax[cuda12_local]"
# Remember "--recursive"! Make Brennan happy.
git clone --recursive git@github.com:shacklettbp/madrona-learn.git
cd madrona-learn
pip install -e .

cd ..

# Remember "--recursive"!
git clone --recursive git@github.com:shacklettbp/madtrade.git
cd madtrade

mkdir build
cmake -S. -Bbuild
make -j build
pip install -e .

bash train.sh simple_run # this should make checkpoints in ckpts/simple_run

```


**Note!** If you've already cloned, you need the run the following every time you pull (if the Madrona submodules have been updated)

```
git submodule update --init
git submodule update --recursive
```

If you don't it may throw compiler errors.
