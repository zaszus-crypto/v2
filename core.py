import numpy as np
import pandas as pd
  shell: /usr/bin/bash -e {0}
  env:
    pythonLocation: /opt/hostedtoolcache/Python/3.11.16/x64
    PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.11.16/x64/lib/pkgconfig
    Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.16/x64
    Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.16/x64
    Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.16/x64
    LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.11.16/x64/lib
    GEMINI_API_KEY: ***
    TELEGRAM_BOT_TOKEN: ***
    TELEGRAM_CHAT_ID: ***
Traceback (most recent call last):
  File "/home/runner/work/v2/v2/main.py", line 36, in <module>
    from core import (
  File "/home/runner/work/v2/v2/core.py", line 1
    python main.py scan
           ^^^^
SyntaxError: invalid syntax
Error: Process completed with exit code 1.
  shell: /usr/bin/bash -e {0}
  env:
    pythonLocation: /opt/hostedtoolcache/Python/3.11.16/x64
    PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.11.16/x64/lib/pkgconfig
    Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.16/x64
    Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.16/x64
    Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.11.16/x64
    LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.11.16/x64/lib
    GEMINI_API_KEY: ***
    TELEGRAM_BOT_TOKEN: ***
    TELEGRAM_CHAT_ID: ***
Traceback (most recent call last):
  File "/home/runner/work/v2/v2/main.py", line 36, in <module>
    from core import (
ImportError: cannot import name 'Calibrator' from 'core' (/home/runner/work/v2/v2/core.py)
Error: Process completed with exit code 1.
