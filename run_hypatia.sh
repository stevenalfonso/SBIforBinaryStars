#!/bin/bash

# ###### Zona de Parámetros de solicitud de recursos a SLURM ############################
#
#SBATCH --job-name=sbi    #Nombre del job
#SBATCH -p gpu                   #Cola a usar, Default=short (Ver colas y límites en /hpcfs/shared/README/partitions.txt)
#SBATCH -N 1                     #Nodos requeridos, Default=1
#SBATCH -n 1                     #Tasks paralelos, recomendado para MPI, Default=1
#SBATCH --cpus-per-task=4        #Cores requeridos por task, recomendado para multi-thread, Default=1
#SBATCH --mem=16000              #Memoria en Mb por CPU, Default=2048
#SBATCH --time=15-00:00:00       #Tiempo máximo de corrida, Default=2 horas
#SBATCH --mail-user=je.alfonso1@uniandes.edu.co
#SBATCH --mail-type=ALL
#SBATCH -o posterior.%j          #Nombre de archivo de salida


########################################################################################

module load anaconda/python3.9
source ../machine/bin/activate 

date=`/bin/date`
echo "Date: "$date

python3 sbi_train.py

deactivate

date=`/bin/date`
echo "Date: "$date
