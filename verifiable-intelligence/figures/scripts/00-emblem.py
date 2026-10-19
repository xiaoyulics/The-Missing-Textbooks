"""The title-page emblem of Verifiable Intelligence.

The title page carries the Hadamard encoding of the number 1985, the year in which interactive
proofs were introduced (Goldwasser, Micali and Rackoff; Babai). The message u = 1985 has eleven
bits, so its Hadamard codeword is the 2048-bit string f(x) = <u, x> mod 2, x = 0, ..., 2047.
In its natural order the codeword is visibly regular (rows of alternating bits), so the page
shows it in an order fixed by a pseudo-random permutation (NumPy's default generator, seed
1985): display position p, counted row by row from the top left in 32 rows of 64, holds bit
perm[p] of the codeword. Three bits are ringed: positions x, y and x XOR y of the codeword, on
which the linearity test of Blum, Luby and Rubinfeld reads f(x), f(y), f(x XOR y) and accepts
because f(x) + f(y) = f(x XOR y) mod 2. The verifier reads three bits of a proof of length 2048.

Run from the volume root:  python3 figures/scripts/00-emblem.py
It prints the rows (pasted into verifiable.sty) and the three ringed positions.
"""
import numpy as np

U = 1985
N_BITS = 11
COLS, ROWS = 64, 32
X, Y = 45, 673          # the two random positions of the linearity test


def f(x: int) -> int:
    """Hadamard codeword of U at position x: parity of the bits that x and U share."""
    return bin(int(x) & U).count("1") % 2


def main() -> None:
    assert U < 2 ** N_BITS and COLS * ROWS == 2 ** N_BITS
    perm = np.random.default_rng(1985).permutation(2 ** N_BITS)
    where = {int(x): p for p, x in enumerate(perm)}
    word = [f(x) for x in range(2 ** N_BITS)]
    # a codeword passes the linearity test on every pair of positions
    assert all(word[x] ^ word[y] == word[x ^ y] for x in range(2048) for y in range(0, 2048, 97))
    for name, x in (("x", X), ("y", Y), ("x XOR y", X ^ Y)):
        p = where[x]
        print(f"{name:8s} = {x:4d}: f = {word[x]}, shown in row {p // COLS}, column {p % COLS}")
    assert word[X] ^ word[Y] == word[X ^ Y]
    print(f"ones: {sum(word)} of {len(word)}")
    print("% rows as displayed, top to bottom")
    for r in range(ROWS):
        print("".join(str(word[perm[r * COLS + c]]) for c in range(COLS)))


if __name__ == "__main__":
    main()
