# corral reference values

`test_corral_reference.py` compares this package's classical correspondence
analysis with the Bioconductor `corral` implementation:

- Package: <https://bioconductor.org/packages/corral>
- Version: `1.20.0` (Bioconductor 3.22)
- Source revision: `0d657a1`
- R version used: `4.5.3`
- IRLBA version used: `2.3.7`

The checked-in values were generated with:

```bash
R_LIBS_USER=/tmp/corral-min-lib Rscript \
    tests/corral_reference/generate_reference.R /path/to/corral
```

The generator sources corral's `checkers.R`, `utils.R`, and `corral.R`
directly, avoiding installation of its unrelated plotting and Bioconductor
object dependencies. The numerical path is corral's own `corral()` function
and its documented IRLBA backend.

The fixture passes a genes-by-cells matrix to corral, as its API expects, and
uses the standardized residual type with no variance-stabilizing transform.
`set.seed()` makes its IRLBA path reproducible. Normal Python tests use only
the checked-in values and do not require R or corral.

The external fixture includes both standard and principal coordinates. The
package stores only principal coordinates; the test derives standard
coordinates by dividing by singular values before comparing with the fixture.
