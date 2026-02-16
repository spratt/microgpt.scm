# microgpt.scm

A translation of Karpathy's microgpt.py into R7RS-small Scheme — a minimal GPT with autograd, training, and inference.

## Running with Chibi Scheme

### Installing Chibi on Fedora

Chibi is not in the Fedora repos. Build from source:
```
git clone https://github.com/ashinn/chibi-scheme.git
cd chibi-scheme
make -j$(nproc)
sudo make install
```

Fedora doesn't include `/usr/local/lib` in the default shared library search path, so add it:
```
echo /usr/local/lib | sudo tee /etc/ld.so.conf.d/local.conf
sudo ldconfig
```

### Running

```
chibi-scheme microgpt.scm
```

## Compiling and running with CHICKEN

### Installing CHICKEN on Fedora

```
sudo dnf install chicken redhat-rpm-config
```

`redhat-rpm-config` is needed because CHICKEN's egg compiler uses Fedora's hardened build flags.

Install required eggs:
```
sudo chicken-install srfi-1 srfi-27 srfi-69 srfi-132 r7rs
```

### Compiling and running

```
csc -R r7rs microgpt.scm -o microgpt && ./microgpt
```
