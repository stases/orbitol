# Third-party notices

The Orbitol code is MIT-licensed (see [LICENSE](LICENSE)). It contains code derived from the projects below; their
notices follow.

## pymatgen

`orbitol/decode_reduce.py` ports lattice routines of pymatgen 2024.11.13 (https://github.com/materialsproject/pymatgen):
`Lattice._calculate_lll`, the Niggli reduction of `Lattice.get_niggli_reduced_lattice`, `Lattice.from_parameters`,
`Lattice.find_all_mappings` and the point order of `find_points_in_spheres`.

```
The MIT License (MIT)
Copyright (c) 2011-2012 MIT & The Regents of the University of California,
through Lawrence Berkeley National Laboratory

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Crystalite

Orbitol builds on Crystalite (https://github.com/joshrosie/crystalite, commit 03603fe). The pinned Crystalite core is
fetched separately and imported; several Orbitol modules, such as the packed trunk in `orbitol/packed.py`, re-implement
parts of it.

```
MIT License

Copyright (c) 2026 Joshua Rosenthal

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
```

## Installed dependencies

The packages in `pyproject.toml` and `uv.lock` are installed separately and keep their own licences; this repository
does not include their code. Among them, ASE 3.27.0 (LGPL-2.1-or-later) provides the FIRE optimizer and Frechet cell
filter of the sequential relaxer (`orbitol/nequip_relax.py`).
