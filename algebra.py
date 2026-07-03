
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

    if np.all(np.isinf(ratios)):
        return None, pivot_col, ratios

    pivot_row = np.argmin(ratios) + 1

    return pivot_row, pivot_col, ratios

def simplex(z, i, max_iter=200):

    n = len(z) - 1 
    m = len(i)

    nomes = (
        [f"x{i+1}" for i in range(n)] +
        [f"s{i+1}" for i in range(m)]
    )


    M = make_tableau(z, i)

    lista = [(M.copy(), None)]

    status = "otimo"
    
    it = 0
    while np.any(M[0, 1:-2] < 0):
        if it > max_iter:
            status = "max_iter"
            break
        
        pivot_row, pivot_col, ratios = find_pivot(M)

        if pivot_row is None:
            status = "ilimitado"
            break
            
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

    return lista, status