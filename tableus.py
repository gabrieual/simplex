import numpy as np
import pandas as pd

def tableau_para_df(M, n, m, mostrar_outras=False, entra_nome=None):
    var_names = [f"x{k+1}" for k in range(n)] + [f"s{k+1}" for k in range(m)]
    cols = var_names + ["Solução"]
    
    # extrai as variáveis e a solução
    dados = M[:, 1:-1].astype(float)
    idx = list(M[:, 0])
    df = pd.DataFrame(dados, columns=cols, index=idx)
    
    basicas = set(idx[1:]) 
    
    # calcula a intersecção para variáveis fora da base
    for j, var in enumerate(var_names):
        if var not in basicas:
            # Mostra a coluna SE for a variável pivô OU SE o toggle estiver ativado
            if (var == entra_nome) or mostrar_outras:
                col_data = dados[1:, j]
                rhs = dados[1:, -1] # A solução
                
                valid = col_data > 1e-9
                ratios = np.where(valid, rhs / np.where(valid, col_data, 1.0), np.nan)
                
                df[f"Int {var}"] = [np.nan] + list(ratios)
            
    return df

def estilizar(df, pivot_row, pivot_col_idx, entra_nome=None):
    def func(_):
        s = pd.DataFrame('', index=df.index, columns=df.columns)
        if pivot_col_idx is not None:
            s.iloc[:, pivot_col_idx] = 'background-color:#bbdefb'
            
            if entra_nome:
                int_col = f"Int {entra_nome}"
                if int_col in df.columns:
                    int_col_idx = df.columns.get_loc(int_col)
                    s.iloc[:, int_col_idx] = 'background-color:#e3f2fd'
                    
        if pivot_row is not None:
            s.iloc[pivot_row, :] = 'background-color:#ffe0b2'
        if pivot_row is not None and pivot_col_idx is not None:
            s.iloc[pivot_row, pivot_col_idx] = 'background-color:#ef5350;color:white;font-weight:bold'
        return s

    return (
        df.style
        .apply(func, axis=None)
        .format(precision=2, na_rep="—") # Substitui NaNs (valores inválidos/infinitos) por um traço limpo
        .set_table_styles([
            {"selector": "th", "props": [("padding", "6px 10px"), ("background-color", "#37474f"), ("color", "white"), ("border", "1px solid #ccc")]},
            {"selector": "td", "props": [("padding", "6px 10px"), ("border", "1px solid #ccc"), ("text-align", "center")]},
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "100%")]},
        ])
    )

def gerar_exemplo(n, m, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.integers(5, 50, size=n).astype(float)
    coef = rng.integers(1, 10, size=(m, n)).astype(float)
    escala = max(n, m) * 25
    rhs = rng.integers(escala, escala * 2, size=m).astype(float)
    return z, coef, rhs