(geneval) root@develop-zdk2koiy-675bbf7666-kpggq:/home/ma-user/work/wx1468559/Bagel-Reca# pip install clip-benchmark
Looking in indexes: http://cmc-cd-mirror.rnd.huawei.com/pypi/simple, https://mirrors.tools.huawei.com/pypi/simple/
Collecting clip-benchmark
  Downloading https://mirrors.tools.huawei.com/pypi/packages/2a/e1/6a5cb1b56918b5b7821b321262f4305ec42d709c833f9b0d6884d9f27d6d/clip_benchmark-1.6.2-py2.py3-none-any.whl (1.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.9/1.9 MB 15.7 MB/s eta 0:00:00
Requirement already satisfied: torch>=1.8.1 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from clip-benchmark) (2.1.2)
Requirement already satisfied: torchvision>=0.8.9 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from clip-benchmark) (0.16.2)
Requirement already satisfied: tqdm>=2 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from clip-benchmark) (4.65.2)
Collecting scikit-learn<2,>=1.0 (from clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/3f/48/6fdd99f5717045f9984616b5c2ec683d6286d30c0ac234563062132b83ab/scikit_learn-1.3.2-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (11.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.1/11.1 MB 5.1 MB/s eta 0:00:00
Requirement already satisfied: open-clip-torch>=0.2.1 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from clip-benchmark) (2.26.1)
Collecting pycocoevalcap (from clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/08/f9/466f289f1628296b5e368940f89e3cfcfb066d15ddc02ff536dc532b1c93/pycocoevalcap-1.2-py3-none-any.whl (104.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.3/104.3 MB 5.6 MB/s eta 0:00:00
Collecting webdataset>=0.2.31 (from clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/8e/84/cf2319c375f4e061f27354685295905dc81105d2a2d2239baaf6f6e73c87/webdataset-0.2.100-py3-none-any.whl (74 kB)
Collecting transformers (from clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/51/51/b87caa939fedf307496e4dbf412f4b909af3d9ca8b189fc3b65c1faa456f/transformers-4.46.3-py3-none-any.whl (10.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 10.0/10.0 MB 82.8 MB/s eta 0:00:00
Requirement already satisfied: regex in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from open-clip-torch>=0.2.1->clip-benchmark) (2024.11.6)
Requirement already satisfied: ftfy in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from open-clip-torch>=0.2.1->clip-benchmark) (6.2.3)
Requirement already satisfied: huggingface-hub in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from open-clip-torch>=0.2.1->clip-benchmark) (0.36.2)
Requirement already satisfied: timm in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from open-clip-torch>=0.2.1->clip-benchmark) (1.0.27)
Requirement already satisfied: numpy<2.0,>=1.17.3 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from scikit-learn<2,>=1.0->clip-benchmark) (1.24.4)
Collecting scipy>=1.5.0 (from scikit-learn<2,>=1.0->clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/69/f0/fb07a9548e48b687b8bf2fa81d71aba9cfc548d365046ca1c791e24db99d/scipy-1.10.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (34.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 34.5/34.5 MB 5.5 MB/s eta 0:00:00
Collecting joblib>=1.1.1 (from scikit-learn<2,>=1.0->clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/91/29/df4b9b42f2be0b623cbd5e2140cafcaa2bef0759a00b7b70104dcfe2fb51/joblib-1.4.2-py3-none-any.whl (301 kB)
Collecting threadpoolctl>=2.0.0 (from scikit-learn<2,>=1.0->clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/4b/2c/ffbf7a134b9ab11a67b0cf0726453cedd9c5043a4fe7a35d1cefa9a1bcfb/threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
Requirement already satisfied: filelock in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (3.14.0)
Requirement already satisfied: typing-extensions in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (4.13.2)
Requirement already satisfied: sympy in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (1.13.3)
Requirement already satisfied: networkx in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (2.8.8)
Requirement already satisfied: jinja2 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (3.1.6)
Requirement already satisfied: fsspec in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (2025.3.1)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (12.1.105)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (12.1.105)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (12.1.105)
Requirement already satisfied: nvidia-cudnn-cu12==8.9.2.26 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (8.9.2.26)
Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (12.1.3.1)
Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (11.0.2.54)
Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (10.3.2.106)
Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (11.4.5.107)
Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (12.1.0.106)
Requirement already satisfied: nvidia-nccl-cu12==2.18.1 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (2.18.1)
Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (12.1.105)
Requirement already satisfied: triton==2.1.0 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torch>=1.8.1->clip-benchmark) (2.1.0)
Requirement already satisfied: nvidia-nvjitlink-cu12 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from nvidia-cusolver-cu12==11.4.5.107->torch>=1.8.1->clip-benchmark) (12.9.86)
Requirement already satisfied: requests in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torchvision>=0.8.9->clip-benchmark) (2.28.2)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from torchvision>=0.8.9->clip-benchmark) (10.4.0)
Collecting braceexpand (from webdataset>=0.2.31->clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/fa/93/e8c04e80e82391a6e51f218ca49720f64236bc824e92152a2633b74cf7ab/braceexpand-0.1.7-py2.py3-none-any.whl (5.9 kB)
Requirement already satisfied: pyyaml in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from webdataset>=0.2.31->clip-benchmark) (6.0.3)
Collecting pycocotools>=2.0.2 (from pycocoevalcap->clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/6c/11/6cb76ebc71388ac17691bc3da76276d1642af30bf9097de9bb5f64c92cfa/pycocotools-2.0.7-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (439 kB)
Requirement already satisfied: packaging>=20.0 in /root/miniconda3/envs/geneval/lib/python3.8/site-packages (from transformers->clip-benchmark) (24.2)
Collecting tokenizers<0.21,>=0.20 (from transformers->clip-benchmark)
  Downloading https://mirrors.tools.huawei.com/pypi/packages/1a/98/0df883ea6201e35e286a97f5fb2a601bfb5b52e4165f7688a76e4553eeec/tokenizers-0.20.4.tar.gz (343 kB)
  Installing build dependencies ... done
  Getting requirements to build wheel ... done
  Preparing metadata (pyproject.toml) ... error
  error: subprocess-exited-with-error
  
  × Preparing metadata (pyproject.toml) did not run successfully.
  │ exit code: 1
  ╰─> [23 lines of output]
          Updating crates.io index
      warning: spurious network error (3 tries remaining): [28] Timeout was reached (Failed to connect to index.crates.io port 443 after 15626 ms: Connection timed out)
      warning: spurious network error (2 tries remaining): [28] Timeout was reached (Failed to connect to index.crates.io port 443 after 15215 ms: Connection timed out)
      warning: spurious network error (1 tries remaining): [28] Timeout was reached (Failed to connect to index.crates.io port 443 after 15214 ms: Connection timed out)
      error: failed to get `env_logger` as a dependency of package `tokenizers-python v0.20.4 (/tmp/pip-install-w9uw8xnd/tokenizers_6add9ca944de40f7b1110a0c2ca3fff0/bindings/python)`
      
      Caused by:
        failed to query replaced source registry `crates-io`
      
      Caused by:
        download of config.json failed
      
      Caused by:
        failed to download from `https://index.crates.io/config.json`
      
      Caused by:
        [28] Timeout was reached (Failed to connect to index.crates.io port 443 after 15211 ms: Connection timed out)
      💥 maturin failed
        Caused by: Cargo metadata failed. Does your crate compile with `cargo build`?
        Caused by: `cargo metadata` exited with an error:
      Error running maturin: Command '['maturin', 'pep517', 'write-dist-info', '--metadata-directory', '/tmp/pip-modern-metadata-ch88ws87', '--interpreter', '/root/miniconda3/envs/geneval/bin/python3.8']' returned non-zero exit status 1.
      Checking for Rust toolchain....
      Running `maturin pep517 write-dist-info --metadata-directory /tmp/pip-modern-metadata-ch88ws87 --interpreter /root/miniconda3/envs/geneval/bin/python3.8`
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.
