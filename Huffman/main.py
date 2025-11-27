"""
===============================================================
 Program : huffman_compression.py
 Author  : Basado en 05_huffman.py de Juan Manuel Ahuactzin
 Date    : 2025-11-24
 Version : 2.0
===============================================================
Description:
    Implementación completa del algoritmo de Huffman para 
    compresión y descompresión de archivos de texto.
    
Dependencies:
    - Python 3.x
    - bitarray: pip install bitarray
    - graphviz: pip install graphviz
    - pickle (estándar)
===============================================================
"""

import heapq
import pickle
from bitarray import bitarray
from collections import Counter
import os
from graphviz import Digraph
import time
import csv
import statistics
import glob

# Class to represent huffman tree


class Node:
    def __init__(self, x, character=""):
        self.data = x
        self.char = character
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.data < other.data

    def __str__(self, level=0, prefix="Root: ", cumchain=""):
        """Devuelve una representación jerárquica del árbol."""
        result = "   " * level + \
            f"{prefix}({self.char}:{self.data}):{cumchain}\n"
        if self.left:
            result += self.left.__str__(level + 1,
                                        prefix="L-0- ", cumchain=cumchain+'0')
        if self.right:
            result += self.right.__str__(level + 1,
                                         prefix="R-1- ", cumchain=cumchain+'1')
        return result

# Function to traverse tree in preorder
# manner and push the huffman representation
# of each character.


def preOrder(root, ans, curr):
    if root is None:
        return

    # Leaf node represents a character.
    if root.left is None and root.right is None:
        ans[root.char] = curr
        return

    preOrder(root.left, ans, curr + '0')
    preOrder(root.right, ans, curr + '1')


def huffmanCodes(s, freq):
    # Code here
    n = len(s)

    # Min heap for node class.
    pq = []
    for i in range(n):
        tmp = Node(freq[i], s[i])
        heapq.heappush(pq, tmp)

    # Caso especial: un solo carácter
    if len(pq) == 1:
        root = heapq.heappop(pq)
        codes_dic = {root.char: '0'}
        return codes_dic, root

    # Construct huffman tree.
    while len(pq) >= 2:
        # Left node
        l = heapq.heappop(pq)

        # Right node
        r = heapq.heappop(pq)

        newNode = Node(l.data + r.data)
        newNode.left = l
        newNode.right = r

        heapq.heappush(pq, newNode)

    root = heapq.heappop(pq)
    codes_dic = {}
    preOrder(root, codes_dic, "")
    return codes_dic, root


def generate_frequency_stats(text):
    """Genera estadísticas de frecuencia de caracteres en el texto."""
    freq_dict = Counter(text)
    characters = list(freq_dict.keys())
    frequencies = list(freq_dict.values())
    return characters, frequencies, freq_dict


def calculate_average_bits(codes_dic, freq_dict, total_chars):
    """Calcula el número promedio de bits por símbolo."""
    total_bits = 0
    for char, code in codes_dic.items():
        freq = freq_dict.get(char, 0)
        total_bits += len(code) * freq
    avg_bits = total_bits / total_chars if total_chars > 0 else 0
    return avg_bits


def print_huffman_table(codes_dic, freq_dict):
    """Imprime una tabla con caracteres, frecuencias y códigos."""
    print("\n" + "="*60)
    print("TABLA DE CÓDIGOS DE HUFFMAN")
    print("="*60)
    print(f"{'Carácter':<15} {'Frecuencia':<15} {'Código':<20}")
    print("-"*60)
    sorted_items = sorted(codes_dic.items(), key=lambda x: (len(x[1]), x[0]))
    for char, code in sorted_items:
        char_repr = repr(char) if char in ['\n', '\t', ' '] else char
        freq = freq_dict.get(char, 0)
        print(f"{char_repr:<15} {freq:<15} {code:<20}")
    print("="*60)


def visualize_huffman_tree(root, output_file="huffman_tree"):
    """Visualiza el árbol de Huffman usando Graphviz."""

    dot = Digraph(comment='Árbol de Huffman')
    dot.attr(rankdir='TB')
    dot.attr('node', shape='circle', style='filled', fillcolor='lightblue')

    node_counter = [0]

    def add_nodes_edges(node, parent_id=None, edge_label=""):
        if node is None:
            return

        if node.char:  # Nodo hoja
            node_id = f"leaf_{node.char}_{node.data}"
            label = f"{repr(node.char) if node.char in ['\\n', '\\t', ' '] else node.char}\\n{node.data}"
            dot.node(node_id, label, fillcolor='lightgreen')
        else:  # Nodo interno
            node_id = f"internal_{node_counter[0]}"
            node_counter[0] += 1
            label = str(node.data)
            dot.node(node_id, label, fillcolor='lightcoral')

        if parent_id is not None:
            dot.edge(parent_id, node_id, label=edge_label)

        if node.left:
            add_nodes_edges(node.left, node_id, "0")
        if node.right:
            add_nodes_edges(node.right, node_id, "1")

        return node_id

    add_nodes_edges(root)

    try:
        dot.render(output_file, format='png', cleanup=True)
        print(f"\nAr bol visualizado guardado como: {output_file}.png")
    except Exception as e:
        print(f"Error al generar visualización: {e}")
        with open(f"{output_file}.dot", 'w') as f:
            f.write(dot.source)
        print(f"Código DOT guardado como: {output_file}.dot")


def encode_file(input_filename, output_filename=None):
    """Comprime un archivo de texto usando el algoritmo de Huffman."""
    if output_filename is None:
        output_filename = input_filename + ".huff"

    # Leer archivo original
    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return None

    if not text:
        print("Error: el archivo está vacío")
        return None

    # Generar estadísticas
    characters, frequencies, freq_dict = generate_frequency_stats(text)

    # Construir árbol y códigos
    codes_dic, root = huffmanCodes(characters, frequencies)

    # Crear códigos como bitarray
    codes = {}
    for c in codes_dic:
        codes[c] = bitarray(codes_dic[c])

    # Codificar el texto
    encoded_bits = bitarray()
    for char in text:
        encoded_bits.extend(codes[char])

    # Calcular padding
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits.extend([0] * padding)

    # Guardar archivo comprimido
    with open(output_filename, 'wb') as f:
        # Header con pickle: diccionario de códigos, padding y árbol
        header = {
            'codes_dic': codes_dic,
            'padding': padding,
            'tree': root
        }
        pickle.dump(header, f)
        # Escribir bits comprimidos
        encoded_bits.tofile(f)

    # Calcular estadísticas
    original_size = os.path.getsize(input_filename)
    compressed_size = os.path.getsize(output_filename)
    compression_factor = compressed_size / original_size
    avg_bits = calculate_average_bits(codes_dic, freq_dict, len(text))

    stats = {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_factor': compression_factor,
        'avg_bits_per_symbol': avg_bits,
        'codes_dic': codes_dic,
        'freq_dict': freq_dict
    }

    print(f"\nArchivo comprimido: {output_filename}")
    print(f"  Tamaño original: {original_size} bytes")
    print(f"  Tamaño comprimido: {compressed_size} bytes")
    print(f"  Factor de compresión: {compression_factor:.4f}")
    print(f"  Bits promedio por símbolo: {avg_bits:.4f}")

    return stats

def read_huff_file(input_filename):
    #Lee el header y los bits desde el .huff
    with open(input_filename, 'rb') as f:
        header = pickle.load(f)
        bits = bitarray()
        bits.fromfile(f)
    return header, bits

def decode_file_inverse(input_filename, output_filename=None):
    #Decodifica usando diccionario inverso
    if output_filename is None:
        output_filename = input_filename.replace('.huff', '.dec.txt')

    header, bits = read_huff_file(input_filename)
    if header is None:
        return False

    codes_dic = header.get('codes_dic', {})
    padding = header.get('padding', 0)

    if padding:
        bits = bits[:len(bits) - padding]

    # construir diccionario inverso
    inv = {v: k for k, v in codes_dic.items()}

    out_chars = []
    cur = ""
    for b in bits:
        cur += '1' if b else '0'
        if cur in inv:
            out_chars.append(inv[cur])
            cur = ""
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(''.join(out_chars))
    return True

def decode_file_tree(input_filename, output_filename=None):
    """Decodifica bit a bit usando el árbol Huffman"""
    if output_filename is None:
        output_filename = input_filename.replace('.huff', '.dec.txt')

    header, bits = read_huff_file(input_filename)
    if header is None:
        return False

    root = header.get('tree', None)
    padding = header.get('padding', 0)

    if root is None:
        print("Error: header no contiene árbol")
        return False

    if padding:
        bits = bits[:len(bits) - padding]

    # caso especial: arbol con un solo carácter
    if root.left is None and root.right is None:
        single_char = root.char
        count = len(bits)
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(single_char * count)
        return True

    out_chars = []
    node = root
    for b in bits:
        node = node.right if b else node.left
        if node.left is None and node.right is None:
            out_chars.append(node.char)
            node = root

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(''.join(out_chars))
    return True

def find_txt_files(base_dir="../libros"):
    """
      - ../libros/es/*.txt, ../libros/fr/*.txt, ../libros/en/*.txt
    """
    files = []
    best_langs = ['es','fr','en']
    # buscar subcarpetas
    found_any = False
    for lang in best_langs:
        pattern = os.path.join(base_dir, lang, '*.txt')
        matched = glob.glob(pattern)
        if matched:
            found_any = True
            for p in matched:
                files.append((p, lang))
    if found_any:
        return files
    return files

def compress_many(file_list, out_dir=None):
    results = []
    for path, language in file_list:
        base = os.path.basename(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            outname = os.path.join(out_dir, base + '.huff')
        else:
            outname = path + '.huff'
        stats = encode_file(path, outname)
        if stats:
            results.append({
                'filename': base,
                'language': language,
                'original_size': stats['original_size'],
                'compressed_size': stats['compressed_size'],
                'compression_factor': stats['compression_factor'],
                'avg_bits_per_symbol': stats['avg_bits_per_symbol'],
                'huff_path': outname,
            })
    return results

def save_csv_stats(rows, csv_path):
    if not rows:
        print("No hay filas")
        return
    keys = list(rows[0].keys())
    keys = [k for k in keys if k != 'tree_root']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in keys}
            writer.writerow(row)

def summarize_by_language(rows):
    by_lang = {}
    for r in rows:
        lang = r.get('language', 'unknown')
        by_lang.setdefault(lang, []).append(r)
    summary = []
    for lang, items in by_lang.items():
        mean_comp = statistics.mean([it['compression_factor'] for it in items]) if items else 0
        mean_bits = statistics.mean([it['avg_bits_per_symbol'] for it in items]) if items else 0
        summary.append({
            'language': lang,
            'avg_compression_factor': mean_comp,
            'avg_bits_per_symbol': mean_bits,
            'files_count': len(items)
        })
    return summary

def measure_decoder(decoder_func, input_huff, runs=500):
    times = []
    tmp_out = input_huff.replace('.huff', '.tmp.dec.txt')
    for _ in range(runs):
        t0 = time.perf_counter()
        ok = decoder_func(input_huff, tmp_out)
        t1 = time.perf_counter()
        if not ok:
            raise RuntimeError("Error en decodificador durante la medición")
        times.append(t1 - t0)
    try:
        os.remove(tmp_out)
    except Exception:
        pass
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, stdev

def main():
    base_dir = "./libros"
    out_huff_dir = "./huff_out"
    os.makedirs(out_huff_dir, exist_ok=True)

    files = find_txt_files(base_dir)
    
    per_file_rows = compress_many(files, out_dir=out_huff_dir)

    save_csv_stats(per_file_rows, "stats_per_file.csv")

    # Resumen por idioma
    summary = summarize_by_language(per_file_rows)
    with open("stats_by_language.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['language','avg_compression_factor','avg_bits_per_symbol','files_count'])
        writer.writeheader()
        for s in summary:
            writer.writerow(s)

    #Crear visualizaciones
    examples = {}
    for r in per_file_rows:
        lang = r['language']
        if lang not in examples:
            examples[lang] = r
    for lang, row in examples.items():
        tree_root = row.get('tree_root')
        if tree_root is None:
            huff_path = row.get('huff_path')
            header, _ = read_huff_file(huff_path)
            tree_root = header.get('tree') if header else None
        if tree_root:
            safe_name = f"tree_{lang}"
            visualize_huffman_tree(tree_root, output_file=safe_name)

    #Medir y sacar promedio
    timing_rows = []
    runs = 500
    for lang, row in examples.items():
        huff = row.get('huff_path')
        if not huff:
            continue
        try:
            m_inv, s_inv = measure_decoder(decode_file_inverse, huff, runs)
            m_tree, s_tree = measure_decoder(decode_file_tree, huff, runs)
            timing_rows.append({
                'language': lang,
                'huff_file': os.path.basename(huff),
                'decoder': 'inverse',
                'runs': runs,
                'mean_seconds': m_inv,
                'stdev_seconds': s_inv
            })
            timing_rows.append({
                'language': lang,
                'huff_file': os.path.basename(huff),
                'decoder': 'tree',
                'runs': runs,
                'mean_seconds': m_tree,
                'stdev_seconds': s_tree
            })
            print(f"  inverse: mean={m_inv:.6f}s std={s_inv:.6f}s")
            print(f"  tree:    mean={m_tree:.6f}s std={s_tree:.6f}s")
        except Exception as e:
            print(f"Error midiendo tiempos para {huff}: {e}")

    if timing_rows:
        keys = list(timing_rows[0].keys())
        with open("timings.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in timing_rows:
                writer.writerow(r)


main()
