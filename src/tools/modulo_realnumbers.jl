export modfloat

# A function that returns true if a float number a is an integer multiple of a float b

modfloat(a, b) = abs(mod(a + 0.5 * b, b) - 0.5 * b) < 1e-12
