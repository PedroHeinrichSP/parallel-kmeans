# parallel-kmeans

Implementação de K-Means como projeto da disciplina Computação Paralela do curso de Ciência da Computação na PUC Minas.
Baseado na implementação de [Lakhan Nad](https://github.com/TheAlgorithms/C/blob/master/machine_learning/k_means_clustering.c).

## Requisitos

- `gcc`
- `bash`
- `nvcc`, para compilar a implementação CUDA
- ImageMagick, para gerar a imagem PNG a partir do EPS (`convert`)

## Compilação

Use o script de compilação:

```sh
./scripts/compile.sh
```

O executável será criado em:

```txt
artifacts/executables/kmeans_seq
```

## Execução

Use o script de execução:

```sh
./scripts/run.sh
```

Por padrão, o script executa o K-Means sequencial com `1000000` observações e `5` clusters.

Também é possível informar esses valores:

```sh
./scripts/run.sh <quantidade_de_observacoes> <quantidade_de_clusters> <algoritmo_usado>
```

O terceiro argumento seleciona a implementação:

```txt
c  CUDA
o  OpenMP
g  OpenMP GPU
```

Exemplo:

```sh
./scripts/run.sh 10000 4
```

Para executar a versão CUDA:

```sh
./scripts/run.sh 10000 4 c
```

## Saídas

A execução gera os arquivos:

```txt
artifacts/images/eps/image.eps
artifacts/images/png/image.png
```

O arquivo EPS é criado diretamente pelo método `printEPS`. O PNG é gerado pelo
script `run.sh` usando o ImageMagick.
