import numpy as np
import pandas as pd
import streamlit as st
from algebra import simplex
from tableus import *

st.set_page_config(page_title="Simplex", layout="wide")

st.title("Simplex ")

def carregar_oleo_e_gas():
    z = np.array([40, 30, 50, 10, 50, 40, 60, 20, 60, 50, 70, 30], dtype=float)
    
    coef = np.array([
        [ 1,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0], 
        [ 0,  0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0], 
        [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1], 
        [ 1,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0], 
        [ 0,  1,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0], 
        [ 0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  1,  0], 
        [ 0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  1], 
        [-0.5, 0,  0,  0, 0.5, 0,  0,  0, 0.5, 0,  0,  0],
        [ 0,-1.5,  0,  0,  0,-0.5, 0,  0,  0, 1.5, 0,  0],
        [ 0,  0,-0.3,  0,  0,  0,-0.3, 0,  0,  0, 0.7, 0]
    ], dtype=float)
    
    rhs = np.array([10000, 8000, 15000, 12000, 10000, 8000, 10000, 0, 0, 0], dtype=float)
    
    st.session_state["z_vals"] = z
    st.session_state["coef_vals"] = coef
    st.session_state["rhs_vals"] = rhs
    st.session_state.pop("lista", None)


if "z_vals" not in st.session_state:
    carregar_oleo_e_gas()

with st.sidebar:
    st.header("Definição do problema")
    n = st.slider("Número de variáveis de decisão (x)", min_value=2, max_value=15, value=st.session_state["z_vals"].shape[0])
    m = st.slider("Número de restrições", min_value=1, max_value=15, value=st.session_state["rhs_vals"].shape[0])

    if st.button("Gerar exemplo aleatório", use_container_width=True):
        z_ex, coef_ex, rhs_ex = gerar_exemplo(n, m)
        st.session_state["z_vals"] = z_ex
        st.session_state["coef_vals"] = coef_ex
        st.session_state["rhs_vals"] = rhs_ex
        st.session_state.pop("lista", None)
        st.rerun()
        
    if st.button("Recarregar Óleo e Gás", use_container_width=True):
        carregar_oleo_e_gas()
        st.rerun()

if st.session_state["z_vals"].shape[0] != n or st.session_state["coef_vals"].shape != (m, n):
    z0, coef0, rhs0 = gerar_exemplo(n, m)
    st.session_state["z_vals"] = z0
    st.session_state["coef_vals"] = coef0
    st.session_state["rhs_vals"] = rhs0

var_cols = [f"x{k+1}" for k in range(n)]

st.subheader("Função objetivo")
obj_df = pd.DataFrame([st.session_state["z_vals"]], columns=var_cols)
obj_edit = st.data_editor(obj_df, hide_index=True, use_container_width=True, key="obj_editor")

st.subheader("Restrições")
restr_df = pd.DataFrame(
    np.hstack([st.session_state["coef_vals"], st.session_state["rhs_vals"].reshape(-1, 1)]),
    columns=var_cols + ["RHS"],
)
restr_edit = st.data_editor(restr_df, hide_index=True, use_container_width=True, key="restr_editor")

col_resolver, col_reset = st.columns([1, 1])
with col_resolver:
    resolver = st.button("▶ Resolver", type="primary", use_container_width=True)
with col_reset:
    if st.button("Limpar solução", use_container_width=True):
        st.session_state.pop("lista", None)
        st.rerun()

if resolver:
    z_vals = obj_edit.to_numpy(dtype=float)[0]
    coef_vals = restr_edit.to_numpy(dtype=float)[:, :n]
    rhs_vals = restr_edit.to_numpy(dtype=float)[:, n]

    if np.any(rhs_vals < 0):
        st.error("⚠️ Este Simplex (forma padrão simples) exige RHS ≥ 0 em todas as restrições. Ajuste os valores da coluna RHS.")
    else:
        z_full = np.append(z_vals, 0.0)
        i_full = np.hstack([coef_vals, rhs_vals.reshape(-1, 1)])
        lista, status = simplex(z_full, i_full)
        st.session_state["lista"] = lista
        st.session_state["status"] = status
        st.session_state["n"] = n
        st.session_state["m"] = m
        st.session_state["step"] = 0

# =============================================================================
# NAVEGAÇÃO PASSO A PASSO
# =============================================================================

if "lista" in st.session_state:
    st.divider()
    st.subheader("Passo a passo do Simplex")

    lista = st.session_state["lista"]
    status = st.session_state["status"]
    n_s = st.session_state["n"]
    m_s = st.session_state["m"]
    total = len(lista)

    if "step" not in st.session_state:
        st.session_state["step"] = 0

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if st.button("←", use_container_width=True, disabled=st.session_state["step"] == 0):
            st.session_state["step"] -= 1
            st.rerun()
    with c3:
        if st.button("→", use_container_width=True, disabled=st.session_state["step"] == total - 1):
            st.session_state["step"] += 1
            st.rerun()
    with c2:
        novo = st.slider("Passo", min_value=0, max_value=total - 1, value=st.session_state["step"], key="slider_passo")
        if novo != st.session_state["step"]:
            st.session_state["step"] = novo
            st.rerun()

    step = st.session_state["step"]
    M, pivot_aplicado = lista[step]

    st.markdown("<br>", unsafe_allow_html=True)

    mostrar_outras = st.toggle("Mostrar intersecção das demais variáveis", value=False)

    pivot_row = pivot_col_idx = None
    entra_nome = sai_nome = None
    pivot_valor = None
    
    if step < total - 1:
        _, prox_pivot = lista[step + 1]
        pivot_row, pivot_col = prox_pivot
        pivot_col_idx = pivot_col - 1
        
        var_names = [f"x{k+1}" for k in range(n_s)] + [f"s{k+1}" for k in range(m_s)]
        entra_nome = var_names[pivot_col_idx]
        
        sai_nome = M[pivot_row, 0] 
        pivot_valor = float(M[pivot_row, pivot_col])

    # Passa o estado do toggle e o nome da variável que entra
    df = tableau_para_df(M, n_s, m_s, mostrar_outras=mostrar_outras, entra_nome=entra_nome)

    st.markdown(f"**Iteração {step} de {total - 1}**")

    styler = estilizar(df, pivot_row, pivot_col_idx, entra_nome)
    st.markdown(styler.to_html(), unsafe_allow_html=True)

    st.markdown("")
    legenda = ("🔵 entra na base &nbsp;&nbsp;|&nbsp;&nbsp; 🟠 sai da base &nbsp;&nbsp;|&nbsp;&nbsp; 🔴 elemento pivô")
    st.caption(legenda)

    if pivot_row is not None:
        st.info(f"No próximo passo: **{entra_nome}** entra na base e **{sai_nome}** sai. Elemento pivô = **{pivot_valor:.2f}**.")
    else:
        z_atual = float(M[0, -2]) 
        if status == "otimo":
            st.success(f"Solução ótima encontrada! Z = {z_atual:.2f}")
            valores = {}
            for row_idx in range(1, M.shape[0]):
                nome = M[row_idx, 0]
                if nome.startswith("x"):
                    valores[nome] = float(M[row_idx, -2]) 
            for k in range(n_s):
                valores.setdefault(f"x{k+1}", 0.0)
            resumo = pd.DataFrame({"Variável": list(valores.keys()), "Valor": list(valores.values())}).sort_values("Variável")
            st.dataframe(resumo, hide_index=True, use_container_width=True)
        elif status == "ilimitado":
            st.error("Problema ilimitado, Z pode crescer indefinidamente.")
        elif status == "max_iter":
            st.warning("Número máximo de iterações atingido.")