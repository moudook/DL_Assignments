import sys

def read_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def write_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def generate_node_surfaces(dataset_type, task, best_arch, layers_config, caption_prefix):
    lines = []
    nodes_list = []
    for layer_path, layer_disp, n_nodes in layers_config:
        for i in range(n_nodes):
            nodes_list.append((layer_path, layer_disp, i))
    
    for chunk_idx in range(0, len(nodes_list), 4):
        chunk = nodes_list[chunk_idx:chunk_idx+4]
        lines.append(r'\begin{figure}[H]')
        lines.append(r'    \centering')
        for idx, (l_path, l_disp, n_idx) in enumerate(chunk):
            for split, split_disp in [('train', 'Train'), ('val', 'Val'), ('test', 'Test')]:
                prefix = "curve" if dataset_type == "univariate" else "surface"
                img_path = f"outputs/{task}/{dataset_type}/best/{split}/{prefix}_{l_path}_n{n_idx}.png"
                lines.append(r'    \begin{subfigure}[b]{0.32\textwidth}')
                lines.append(rf'        \includegraphics[width=\textwidth]{{{img_path}}}')
                lines.append(rf'        \caption{{{l_disp}-N{n_idx} ({split_disp})}}')
                lines.append(r'    \end{subfigure}')
            if idx < len(chunk) - 1:
                lines.append(r'    \vspace{0.3cm}')
        
        part_text = f" (Part {chunk_idx//4 + 1})" if len(nodes_list) > 4 else ""
        lines.append(rf'    \caption{{{caption_prefix}: Node outputs{part_text}.}}')
        lines.append(r'\end{figure}')
        lines.append('')
    return '\n'.join(lines)


v9 = read_file('Group11_Assignment2_report_v9.tex')

# 1. LS Decision Regions
dr_ls_old = r"""\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \includegraphics[width=\textwidth]{outputs/classification/ls/best/decision_regions_train.png}
        \caption{Training}
    \end{subfigure}
    \begin{subfigure}[b]{0.32\textwidth}
        \includegraphics[width=\textwidth]{outputs/classification/ls/best/decision_regions_val.png}
        \caption{Validation}
    \end{subfigure}
    \begin{subfigure}[b]{0.32\textwidth}
        \includegraphics[width=\textwidth]{outputs/classification/ls/best/decision_regions_test.png}
        \caption{Test}
    \end{subfigure}
    \caption{Dataset 1 (LS): decision regions --- best architecture (1HL$\times$2) across train/val/test.}
\end{figure}"""

dr_ls_new = r"""\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{outputs/classification/ls/best/decision_regions_train.png}
    \caption{Dataset 1 (LS): decision regions --- best architecture (1HL$\times$2) on training data.}
\end{figure}"""
v10 = v9.replace(dr_ls_old, dr_ls_new)

# 2. NLS Decision Regions
dr_nls_old = r"""\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \includegraphics[width=\textwidth]{outputs/classification/nls/best/decision_regions_train.png}
        \caption{Training}
    \end{subfigure}
    \begin{subfigure}[b]{0.32\textwidth}
        \includegraphics[width=\textwidth]{outputs/classification/nls/best/decision_regions_val.png}
        \caption{Validation}
    \end{subfigure}
    \begin{subfigure}[b]{0.32\textwidth}
        \includegraphics[width=\textwidth]{outputs/classification/nls/best/decision_regions_test.png}
        \caption{Test}
    \end{subfigure}
    \caption{Dataset 2 (NLS): decision regions --- 2HL $16\times8$. The network learns concentric non-linear boundaries separating all three ring-structured classes.}
\end{figure}"""

dr_nls_new = r"""\begin{figure}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{outputs/classification/nls/best/decision_regions_train.png}
    \caption{Dataset 2 (NLS): decision regions --- best architecture (2HL $16\times8$) on training data. The network learns concentric non-linear boundaries separating all three ring-structured classes.}
\end{figure}"""
v10 = v10.replace(dr_nls_old, dr_nls_new)

# 3. LS Classification Node Surfaces
ls_start = r"\subsubsection*{Node Output Surfaces --- All Nodes, Best Architecture (1HL$\times$2)}"
ls_end = r"\subsubsection*{Analysis and Inferences}"
idx_start = v10.find(ls_start)
idx_end = v10.find(ls_end)
if idx_start != -1 and idx_end != -1:
    v10 = v10[:idx_start] + ls_start + "\n\n" + generate_node_surfaces(
        "ls", "classification", "1HL$\\times$2",
        [("hidden_l1", "H1", 2), ("output_l2", "Out", 3)],
        "Dataset 1 (LS) 1HL$\\times$2"
    ) + "\n" + v10[idx_end:]

# 4. NLS Classification Node Surfaces
nls_start = r"\subsubsection*{Node Output Surfaces --- Best Architecture (2HL $16\times8$)}"
nls_end = r"\subsubsection*{Analysis and Inferences}"
idx_start = v10.find(nls_start)
idx_end = v10.find(nls_end, idx_start)
if idx_start != -1 and idx_end != -1:
    new_heading = r"\subsubsection*{Node Output Surfaces --- All Nodes, Best Architecture (2HL $16\times8$)}"
    v10 = v10[:idx_start] + new_heading + "\n\n" + generate_node_surfaces(
        "nls", "classification", "2HL $16\\times8$",
        [("hidden_l1", "H1", 16), ("hidden_l2", "H2", 8), ("output_l3", "Out", 3)],
        "Dataset 2 (NLS) 2HL $16\\times8$"
    ) + "\n" + v10[idx_end:]

# 5. Univariate Regression Node Surfaces
uni_start = r"\subsubsection*{Node Output Curves --- All Nodes, Best Architecture (1HL$\times$2)}"
uni_end = r"%==========================================================================="
idx_start = v10.find(uni_start)
idx_end = v10.find(uni_end, idx_start)
if idx_start != -1 and idx_end != -1:
    v10 = v10[:idx_start] + uni_start + "\n\n" + generate_node_surfaces(
        "univariate", "regression", "1HL$\\times$2",
        [("hidden_l1", "H1", 2), ("output_l2", "Out", 1)],
        "Univariate Regression 1HL$\\times$2"
    ) + "\n" + v10[idx_end:]

# 6. Bivariate Regression Node Surfaces (1HL and 2HL)
biv_start = r"\subsubsection*{Node Output Surfaces --- Best Architectures}"
biv_end = r"\subsection{Comparison with Assignment 1: Regression}"
idx_start = v10.find(biv_start)
idx_end = v10.find(biv_end, idx_start)
if idx_start != -1 and idx_end != -1:
    new_heading = r"\subsubsection*{Node Output Surfaces --- All Nodes, Best Architectures (1HL and 2HL)}"
    s1 = generate_node_surfaces(
        "bivariate/2hl", "regression", "2HL $16\\times8$",
        [("hidden_l1", "H1", 16), ("hidden_l2", "H2", 8), ("output_l3", "Out", 1)],
        "Bivariate Regression 2HL $16\\times8$"
    )
    s2 = generate_node_surfaces(
        "bivariate/1hl", "regression", "1HL$\\times$16",
        [("hidden_l1", "H1", 16), ("output_l2", "Out", 1)],
        "Bivariate Regression 1HL$\\times$16"
    )
    v10 = v10[:idx_start] + new_heading + "\n\n" + s1 + "\n" + s2 + "\n" + v10[idx_end:]

# 7. Remove the Appendix
idx_app = v10.find(r"\appendix")
if idx_app != -1:
    v10 = v10[:idx_app] + r"\end{document}" + "\n"

write_file('Group11_Assignment2_report_v10.tex', v10)
print("Successfully wrote Group11_Assignment2_report_v10.tex")
