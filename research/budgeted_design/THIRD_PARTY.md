# Source and license

Task, sequential contrastive objective, and `dad_native` architecture adapted
from [ae-foster/dad](https://github.com/ae-foster/dad), commit
`4b1008174e1531d1f14601d83cef481c0f586f36`.
Relevant files: `location_finding.py`, `neural/modules.py`, `contrastive/mi.py`,
`location_finding_eval.py`, `README.md`. The optional author-source regression
test executes selected unmodified Torch definitions from that checkout.

This is a Torch-only reimplementation, **not an execution of the original
Pyro 1.6/MLflow training stack or a reproduction of the published scores**.
No upstream files are modified. The current environment uses Torch 2.7.1 CUDA
12.8; no packages installed into the old FPL/UAV environment.

## MIT License

Copyright (c) 2021 Adam Foster, Desi R. Ivanova

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
