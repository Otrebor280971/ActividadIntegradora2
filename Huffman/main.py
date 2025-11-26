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
        print(f"\n✓ Árbol visualizado guardado como: {output_file}.png")
    except Exception as e:
        print(f"Error al generar visualización: {e}")
        with open(f"{output_file}.dot", 'w') as f:
            f.write(dot.source)
        print(f"✓ Código DOT guardado como: {output_file}.dot")


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

    print(f"\n✓ Archivo comprimido: {output_filename}")
    print(f"  Tamaño original: {original_size} bytes")
    print(f"  Tamaño comprimido: {compressed_size} bytes")
    print(f"  Factor de compresión: {compression_factor:.4f}")
    print(f"  Bits promedio por símbolo: {avg_bits:.4f}")

    return stats
