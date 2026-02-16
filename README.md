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

## Benchmark

| Implementation | User time | Wall time | vs Python |
|---|---|---|---|
| SBCL (Common Lisp) | 12s | 15s | 0.14x |
| Python 3 | 90s | 1m31s | 1.0x |
| Clojure (JVM) | 129s | 1m55s | 1.4x |
| **CHICKEN (compiled Scheme)** | **170s** | **2m54s** | **1.9x** |
| **Chibi Scheme (interpreted)** | **319s** | **5m21s** | **3.5x** |

CHICKEN compiles Scheme to C, roughly twice as fast as Chibi's interpreter. Both are slower than CPython due to additional allocation overhead — every autograd operation creates new cons cells and lists.
