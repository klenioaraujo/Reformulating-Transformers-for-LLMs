# Appendix

## Core Equations (ASCII Math)

For simplified representation and implementation in computational algebra systems, the core equations are listed below.

```
1. Psi_QRH(r,t) = R * F^-1 { F(k) * F { Psi(r,t) } }

2. F(k) = exp( i * alpha * arctan( ln( |k| + 1e-10 ) ) )

3. R = [ cos(theta/2), sin(theta/2), sin(omega/2), sin(phi/2) ]

4. Hamilton Product: q1 * q2 = 
   [ w1*w2 - x1*x2 - y1*y2 - z1*z2,  // real
     w1*x2 + x1*w2 + y1*z2 - z1*y2,  // i
     w1*y2 - x1*z2 + y1*w2 + z1*x2,  // j
     w1*z2 + x1*y2 - y1*x2 + z1*w2 ] // k

5. Golay Encoding: 
   24 complex coeffs -> 48 floats -> 12-bit message -> 24-bit G24 codeword

6. Leech Mapping: 
   24-bit codeword -> Leech lattice point index
```

## References

### Project Citation
Padilha, Klenio Araujo. (2025). Reformulating Transformers: A Quaternionic-Harmonic Framework. Zenodo. https://doi.org/10.5281/zenodo.17171112

### Cited Works
1.  **Conway, J. H., & Sloane, N. J. A. (1999).** *Sphere Packings, Lattices and Groups*. Springer.
2.  **Thompson, T. M. (1983).** *From Error-Correcting Codes Through Sphere Packings to Simple Groups*. MAA.
3.  **Bao, W., Jin, S., & Markowich, P. A. (2002).** On time-splitting spectral approximations for the Schrödinger equation in the semiclassical regime. *Journal of Computational Physics*.
4.  **Press, W. H., et al. (2007).** *Numerical Recipes*. Cambridge University Press.
5.  **Hardy, G. H., & Wright, E. M. (2008).** *An Introduction to the Theory of Numbers*. Oxford University Press.
