import sys # https://docs.python.org/pt-br/3/library/sys.html
import pickle # https://docs.python.org/3/library/pickle.html
import os # https://docs.python.org/3/library/os.html
import nltk # https://www.nltk.org

# O etiquetador é responsável pelo processo de definição da classe gramatical 
# das palavras, de acordo com as funções sintáticas.        
def criar_etiquetador():
    if os.path.isfile('mac_morpho.pkl'):
        # Carregando um modelo treinado
        input = open('mac_morpho.pkl', 'rb')
        tagger = pickle.load(input)
        input.close()
    else:
        # Obtendo as sentencas etiquetadas do corpus mac_morpho
        tagged_sents = nltk.corpus.mac_morpho.tagged_sents()
        
        # Instanciando etiquetador e treinando com as sentenças etiquetadas
        tagger = nltk.UnigramTagger(tagged_sents)
        # t2 = nltk.BigramTagger(tagged_sents, backoff=t1)    
        # tagger = nltk.TrigramTagger(tagged_sents, backoff=t2)

        # Salvar um modelo treinado em um arquivo para mais tarde usa-lo.
        output = open('mac_morpho.pkl', 'wb')
        pickle.dump(tagger, output, -1) 
        output.close()
    return tagger 

# Assuma que nesses  arquivos  texto, palavras  são separadas por um  ou mais 
# dos seguintes caracteres: espaço em branco ( ), ponto (.), reticências(...) 
# vírgula (,), exclamação (!), interrogação (?) ou enter (\n).
pontuacao = [",", "!", "?", "\n"]

# PREP = preposição,  ART = artigo, KC = conjunção coordenativa, KS = conjução subordinativo
retirarClassificacao = ["PREP", "ART", "KC", "KS"]

# entendimento do significado de um documento.
stopwords = nltk.corpus.stopwords.words("portuguese")

etiquetador = criar_etiquetador()

# A processing interface for removing morphological  affixes from words. This 
# process is known as stemming.
# https://www.nltk.org/api/nltk.stem.html
stemmer = nltk.stem.RSLPStemmer()

# receber  um argumento como  entrada  pela linha  de comando. Este argumento 
# especifica o caminho de um arquivo texto que contém os caminhos de todos os
# arquivos que compõem a base, cada um em uma linha.
caminho = sys.argv[1]

# se caminho nao é arquivo ou nao existe o programa fecha
if not os.path.isfile(caminho):
    exit()

# abre a base informada e le seus documentos
base = open(caminho, 'r')
documentos = base.read().split("\n")
base.close()

# pega o diretorio da base
pastas = caminho.split("/")
diretorio = "".join(pastas[0:len(pastas)-1]) + "/" if len(pastas)-1 != 0 else ""

indicesInvertidos = {}
for numeroArquivo, documento in enumerate(documentos):
    # abre e le o conteudo de todos os documentos
    file = open(diretorio + documento)
    doc = file.read()
    file.close()

    # caracteres ".", ",", "!", "?", "..." e "\n" não devem ser considerados.
    # se o caracter for . troca por um espaço
    semPontuacao = [p if p != '.' else ' ' for p in doc if p not in pontuacao]
    palavras = "".join(semPontuacao).split(" ")

    # stopwords não devem ser levadas em conta na geração do índice invertido
    semStopwords = [p for p in palavras if p not in stopwords]
    semEspacos = [p for p in semStopwords if p not in " "]

    # classificação gramatical
    etiquetados = etiquetador.tag(semEspacos)

    # sem as classificações de preposição, conjunção e artigo
    semClassificacoes = [p[0] for p in etiquetados if p[1] not in retirarClassificacao]
    
    # extrair os radicais das palavras para o índice invertido
    radicais = [stemmer.stem(p) for p in semClassificacoes]
    
    # faz um indice invertido para o documento n
    indice = {p:(numeroArquivo + 1, radicais.count(p)) for p in radicais}

    # junta todos os indices em um só  indice invertido. Queria  muito fazer
    # com compreesão, mas fui incapaz :C
    for chave, valor in indice.items():
        if chave not in indicesInvertidos.keys():
            indicesInvertidos[chave] = []
        indicesInvertidos[chave].append(valor)

# ordena o indice invertido pela chave
indiceInvertidoOrdenado = dict(sorted(indicesInvertidos.items()))

# for chave, valor in indiceInvertidoOrdenado.items():
#     print(f'{chave}:', end="")
#     for v in valor:
#         print(f' {v[0]},{v[1]}', end="")
#     print()

# escreve o indice invertido no arquivo indice.txt
arquivo = open('indice.txt', 'w')
for chave, valor in indiceInvertidoOrdenado.items():
    arquivo.write(f'{chave}:')
    for v in valor:
        arquivo.write(f' {v[0]},{v[1]}')
    arquivo.write('\n')
arquivo.close()
