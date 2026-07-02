import numpy as np


def make_tableau(z, i):
    # Tableau inicial
    M = np.vstack((-z, i))

    # Número de restrições
    m = len(i)

    # Variáveis de folga
    I = np.eye(m)
    # Linha da função objetivo
    I = np.vstack((np.zeros((1, m)), I))

    ratios = np.full((M.shape[0], 1), np.nan)

    # tableau
    M = np.hstack((
        M[:, :-1],
        I,
        M[:, -1:],
        ratios
    ))

    # Variáveis básicas
    basicas = np.array(
        [["Z"]] + [[f"s{i+1}"] for i in range(m)],
        dtype=object
    )

    M = np.hstack((basicas, M.astype(object)))

    return M

def find_pivot(M):

    pivot_col = np.argmin(M[0, 1:-2]) + 1

    col = M[1:, pivot_col].astype(float)
    rhs = M[1:, -2].astype(float)

    ratios = np.where(col > 0, rhs / col, np.inf)

    pivot_row = np.argmin(ratios) + 1

    return pivot_row, pivot_col, ratios

def simplex(z, i):

    n = len(z) - 1 
    m = len(i)

    nomes = (
        [f"x{i+1}" for i in range(n)] +
        [f"s{i+1}" for i in range(m)]
    )


    M = make_tableau(z, i)

    lista = [(M.copy(), None)]


    while np.any(M[0, 1:-2] < 0):

        pivot_row, pivot_col, ratios = find_pivot(M)

        M[pivot_row, 0] = nomes[pivot_col - 1]

        # Atualiza coluna das razões
        M[:, -1] = np.nan
        M[1:, -1] = ratios

        # Normaliza a linha pivô 
        M[pivot_row, 1:] = (
            M[pivot_row, 1:].astype(float) /
            float(M[pivot_row, pivot_col])
        )

        # Eliminação de Gauss
        for r in range(M.shape[0]):
            if r != pivot_row:
                fator = float(M[r, pivot_col])
                M[r, 1:] = (
                    M[r, 1:].astype(float) -
                    fator * M[pivot_row, 1:].astype(float)
                )

        lista.append((M.copy(), (pivot_row, pivot_col)))

    return lista


if __name__ == "__main__":
    # função objetivo
    z = np.array([50, 40, 30, 20, 0], dtype=float)

    # inequações
    i = np.array([
        [2, 1, 1, 1, 100],
        [1, 3, 2, 1, 120],
        [2, 2, 1, 3, 150],
        [1, 1, 2, 2, 90],
    ], dtype=float)

    a = simplex(z, i)

    for idx, (M, pivot) in enumerate(a):
        print(f"I {idx}:")

        M_print = M.copy()
        M_print[:, 1:] = np.round(M_print[:, 1:].astype(float), 2)
        print(M_print)

        if pivot is None:
            print("Pivot: -")
        else:
            print(f"Pivot: ({pivot[0]}, {pivot[1]}) = {M[pivot[0], pivot[1]]}")

        print()