#!/bin/bash

# --- CONFIGURAZIONE GLOBALE ---
EXECUTABLE_PATH="openMP/build/openMP" # Modifica con il percorso corretto
REP=5                                 # Numero di ripetizioni per ogni configurazione
F_VALUE=1                             # Valore fisso per f (terzo parametro dell'eseguibile)

# Nomi dei file CSV di output
CSV_N_SCALING="profiling_N_scaling_omp.csv"
CSV_HARD_SCALING="profiling_hard_scaling_omp.csv"
CSV_SOFT_SCALING="profiling_soft_scaling_omp.csv"

# --- CONTROLLI PRELIMINARI ---
if [ ! -x "$EXECUTABLE_PATH" ]; then
  echo "Errore: Eseguibile '$EXECUTABLE_PATH' non trovato o non eseguibile."
  exit 1
fi

# Funzione per ottenere il numero di processori logici
get_num_cores() {
  if command -v nproc &> /dev/null; then
    nproc
  elif command -v sysctl &> /dev/null && sysctl -n hw.ncpu &> /dev/null; then # macOS
    sysctl -n hw.ncpu
  else
    echo "4" # Valore di fallback
    echo "ATTENZIONE: Impossibile determinare il numero di core. Usato fallback: 4. Considera di impostare THREADS_VALUES manualmente." >&2
  fi
}
NUM_CORES=$(get_num_cores)
echo "INFO: Numero di core/processori logici rilevati: $NUM_CORES"

# Funzione per generare la sequenza di thread: potenze di 2 fino a NUM_CORES e NUM_CORES stesso.
# Assicura che i valori siano >0, <=NUM_CORES, unici, ordinati, e che 1 sia presente.
# Funzione per generare la sequenza di thread: un range da 1 a NUM_CORES.
generate_thread_values() {
  local max_cores="$1"
  local threads_array=()

  if [ "$max_cores" -le 0 ]; then # Gestione caso NUM_CORES non positivo
      echo "1" # Default a 1 se max_cores non è valido
      return
  fi

  for (( i=1; i<=max_cores; i++ )); do
    threads_array+=("$i")
  done
  
  if [ ${#threads_array[@]} -eq 0 ]; then # Fallback se il ciclo non produce nulla (non dovrebbe succedere)
      threads_array=(1)
  fi

  echo "${threads_array[@]}"
}


# --- FUNZIONE HELPER PER ESECUZIONE E LOGGING ---
run_and_log() {
  local N_val="$1"
  local K_val="$2"
  local run_num="$3"
  local set_omp_threads_value="$4" # Valore OMP_NUM_THREADS impostato dallo script
  local csv_file="$5"
  local test_type="$6"
  local current_f_value="$F_VALUE"

  # set_omp_threads_value è il valore numerico di OMP_NUM_THREADS che è stato impostato.
  local log_msg_omp_threads=", OMP_NUM_THREADS(set)=${set_omp_threads_value}"
  echo "    $test_type - Run $run_num/$REP: N=$N_val, K=$K_val${log_msg_omp_threads}"

  local output_programma
  # Assicurati che OMP_NUM_THREADS sia disponibile per il programma eseguito
  # Se export OMP_NUM_THREADS è fatto prima del loop delle ripetizioni, è già nell'ambiente.
  output_programma=$("$EXECUTABLE_PATH" "$N_val" "$K_val" "$current_f_value")
  local exit_code=$?

  # csv_column_omp_threads_val è ciò che viene scritto nel CSV per la colonna SetOMPThreads.
  # Dovrebbe essere sempre il valore di set_omp_threads_value.
  local csv_column_omp_threads_val="$set_omp_threads_value"
  if [ -z "$csv_column_omp_threads_val" ]; then
    # Questo non dovrebbe succedere se lo script chiama run_and_log correttamente.
    csv_column_omp_threads_val="NA_SCRIPT_ERR" # Indica un problema nello script chiamante
    echo "      AVVISO INTERNO: set_omp_threads_value era vuoto per N=$N_val, K=$K_val. Usato '$csv_column_omp_threads_val' nel CSV." >&2
  fi

  if [ $exit_code -ne 0 ]; then
    echo "      ATTENZIONE: Eseguibile (N=$N_val, K=$K_val, OMP_THREADS(set)=$set_omp_threads_value) codice errore $exit_code. Output: '$output_programma'. Riga con NA nel CSV."
    # Struttura CSV unificata: N,K,SetOMPThreads,RunNumber,Time,Throughput,ReportedThreads
    echo "$N_val,$K_val,$csv_column_omp_threads_val,$run_num,NA,NA,NA" >> "$csv_file"
    return 1
  fi

  if [ -z "$output_programma" ]; then
    echo "      ATTENZIONE: Output vuoto (N=$N_val, K=$K_val, OMP_THREADS(set)=$set_omp_threads_value). Riga con NA nel CSV."
    echo "$N_val,$K_val,$csv_column_omp_threads_val,$run_num,NA,NA,NA" >> "$csv_file"
    return 1
  fi

  local num_fields
  num_fields=$(echo "$output_programma" | awk -F',' '{print NF}')
  if [ "$num_fields" -ne 3 ]; then
    echo "      ATTENZIONE: Formato output non valido (N=$N_val, K=$K_val, OMP_THREADS(set)=$set_omp_threads_value). Output: '$output_programma'. Attesi 3 campi, trovati $num_fields. Riga con NA nel CSV."
    echo "$N_val,$K_val,$csv_column_omp_threads_val,$run_num,NA,NA,NA" >> "$csv_file"
    return 1
  fi

  local time_val=$(echo "$output_programma" | cut -d',' -f1)
  local throughput_val=$(echo "$output_programma" | cut -d',' -f2)
  local reported_threads_val=$(echo "$output_programma" | cut -d',' -f3)

  # Struttura CSV unificata (N o N_Calculated, K, SetOMPThreads, RunNumber, Time, Throughput, ReportedThreads)
  echo "$N_val,$K_val,$csv_column_omp_threads_val,$run_num,$time_val,$throughput_val,$reported_threads_val" >> "$csv_file"
  return 0
}

# --- PROFILING 1: VARIAZIONE DI N (Scaling del Problema con specifici OMP_NUM_THREADS) ---
echo "--------------------------------------------------------------"
echo "INIZIO PROFILING 1: Variazione di N e OMP_NUM_THREADS"
echo "--------------------------------------------------------------"
K_N_SCALING=1000
N_VALUES_N_SCALING=()
for i in {16..20}; do # Esempio: 2^16, 2^17, 2^18, 2^19, 2^20
  N_VALUES_N_SCALING+=($((2**i)))
done

# Determina i valori di OMP_NUM_THREADS da testare: 1, NUM_CORES/2, NUM_CORES
# Assicura valori unici e ordinati, e che siano almeno 1.
THREADS_N_SCALING_BASE=()
THREADS_N_SCALING_BASE+=(1) # Sempre testare con 1 thread

if [ "$NUM_CORES" -gt 1 ]; then
    HALF_CORES=$((NUM_CORES / 2))
    if [ "$HALF_CORES" -lt 1 ]; then # Dovrebbe essere sempre >= 1 se NUM_CORES > 1
        HALF_CORES=1
    fi
    THREADS_N_SCALING_BASE+=("$HALF_CORES")
    THREADS_N_SCALING_BASE+=("$NUM_CORES")
else
    # Se NUM_CORES è 1, HALF_CORES e MAX_CORES sono già coperti da 1.
    : # No-op, 1 è già in THREADS_N_SCALING_BASE
fi
# Ottieni valori unici e ordinati
THREADS_FOR_N_SCALING=($(echo "${THREADS_N_SCALING_BASE[@]}" | tr ' ' '\n' | sort -un | tr '\n' ' '))

echo "INFO: N-Scaling - Valori di N da testare: ${N_VALUES_N_SCALING[*]}"
echo "INFO: N-Scaling - Valori di OMP_NUM_THREADS da testare per ogni N: ${THREADS_FOR_N_SCALING[*]}"

# Aggiorna l'header del CSV per includere SetOMPThreads
echo "N,K,SetOMPThreads,RunNumber,Time,Throughput,ReportedThreads" > "$CSV_N_SCALING"

for N_CURRENT in "${N_VALUES_N_SCALING[@]}"; do
  for OMP_THREADS_CURRENT in "${THREADS_FOR_N_SCALING[@]}"; do
    echo "  Profiling N-Scaling: N = $N_CURRENT, K = $K_N_SCALING, OMP_NUM_THREADS = $OMP_THREADS_CURRENT"
    export OMP_NUM_THREADS="$OMP_THREADS_CURRENT"
    for ((r=1; r<=REP; r++)); do
      # Passa il valore corrente di OMP_THREADS a run_and_log
      run_and_log "$N_CURRENT" "$K_N_SCALING" "$r" "$OMP_THREADS_CURRENT" "$CSV_N_SCALING" "N-Scaling"
    done
  done
done
unset OMP_NUM_THREADS # Assicura che OMP_NUM_THREADS sia disimpostato dopo questo blocco di test
echo "Completato Profiling N-Scaling. Risultati in $CSV_N_SCALING"


# --- PROFILING 2: HARD SCALING ---
echo "--------------------------------------------------------------"
echo "INIZIO PROFILING 2: Hard Scaling (N e K fissi, varia OMP_NUM_THREADS)"
echo "--------------------------------------------------------------"
N_HARD_SCALING=$((2**20))
K_HARD_SCALING=1000
THREADS_VALUES_HARD=($(generate_thread_values "$NUM_CORES"))
echo "INFO: Hard Scaling - Valori di OMP_NUM_THREADS da testare: ${THREADS_VALUES_HARD[*]}"

echo "N,K,SetOMPThreads,RunNumber,Time,Throughput,ReportedThreads" > "$CSV_HARD_SCALING"
for OMP_THREADS in "${THREADS_VALUES_HARD[@]}"; do
  echo "  Profiling Hard-Scaling: N = $N_HARD_SCALING, K = $K_HARD_SCALING, OMP_NUM_THREADS = $OMP_THREADS"
  export OMP_NUM_THREADS="$OMP_THREADS"
  for ((r=1; r<=REP; r++)); do
    run_and_log "$N_HARD_SCALING" "$K_HARD_SCALING" "$r" "$OMP_THREADS" "$CSV_HARD_SCALING" "Hard-Scaling"
  done
done
unset OMP_NUM_THREADS
echo "Completato Profiling Hard-Scaling. Risultati in $CSV_HARD_SCALING"


# --- PROFILING 3: SOFT SCALING ---
echo "--------------------------------------------------------------"
echo "INIZIO PROFILING 3: Soft Scaling (K fisso, N scala con OMP_NUM_THREADS)"
echo "--------------------------------------------------------------"
K_SOFT_SCALING=1000
N_BASE_SOFT_SCALING=$((2**16))
THREADS_VALUES_SOFT=($(generate_thread_values "$NUM_CORES"))
echo "INFO: Soft Scaling - Valori di OMP_NUM_THREADS da testare: ${THREADS_VALUES_SOFT[*]}"

echo "N_Calculated,K,SetOMPThreads,RunNumber,Time,Throughput,ReportedThreads" > "$CSV_SOFT_SCALING"
for OMP_THREADS in "${THREADS_VALUES_SOFT[@]}"; do
  N_CURRENT_SOFT=$((N_BASE_SOFT_SCALING * OMP_THREADS))
  echo "  Profiling Soft-Scaling: N_calc = $N_CURRENT_SOFT (Base $N_BASE_SOFT_SCALING * Threads $OMP_THREADS), K = $K_SOFT_SCALING, OMP_NUM_THREADS = $OMP_THREADS"
  export OMP_NUM_THREADS="$OMP_THREADS"
  for ((r=1; r<=REP; r++)); do
    run_and_log "$N_CURRENT_SOFT" "$K_SOFT_SCALING" "$r" "$OMP_THREADS" "$CSV_SOFT_SCALING" "Soft-Scaling"
  done
done
unset OMP_NUM_THREADS
echo "Completato Profiling Soft-Scaling. Risultati in $CSV_SOFT_SCALING"

echo "--------------------------------------------------------------"
echo "TUTTI I PROFILING COMPLETATI."
echo "Controlla i file:"
echo "  - $CSV_N_SCALING"
echo "  - $CSV_HARD_SCALING"
echo "  - $CSV_SOFT_SCALING"
echo "--------------------------------------------------------------"